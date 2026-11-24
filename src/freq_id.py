"""Frequency-domain identification, with coherence.

Why: differentiating in the time domain amplifies noise. In the frequency domain
differentiation is a multiplication by jw, so the frequency response comes
straight out of the cross-spectra. More importantly, **coherence** tells you in
which band the data can be trusted — something no time-domain metric provides.

For each axis it estimates  H(f) = Pxy / Pxx  between commanded torque and
angular rate, with coherence  gamma^2 = |Pxy|^2 / (Pxx * Pyy).

It then fits, only where coherence passes the threshold, the first-order model

    H(jw) = K / (jw + a)

whose time-domain form is  w_dot = K*tau - a*w:  K is control effectiveness and
`a` is damping. The fit is linear in (a, K):

    jw * H(jw)  =  K - a * H(jw)

which gives a damping estimate **independent** of the time-domain regression.
A pure delay is swept on top, and `compare_structures` checks whether a free
pole is the right model at all — for a small multirotor it usually is not.

Usage: python src/freq_id.py data/ulg/<id>/<id>.ulg [--modes Manual] [--coh 0.6]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import coherence, csd, welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import sindy_rates as sr  # noqa: E402

C_A, C_B = "#1F6FB2", "#D1670B"
INK, MUTED, GRID, BG = "#1a1a1a", "#6b6b6b", "#e3e3e0", "#fcfcfb"


def fit_first_order(f, H, w8):
    """Complex least squares of jw*H = K - a*H, weighted by coherence."""
    w = 2 * np.pi * f
    y = 1j * w * H
    A = np.column_stack([-H, np.ones_like(H)])          # [a, K]
    W = np.sqrt(w8)
    Ar = np.vstack([np.real(A) * W[:, None], np.imag(A) * W[:, None]])
    yr = np.concatenate([np.real(y) * W, np.imag(y) * W])
    sol, *_ = np.linalg.lstsq(Ar, yr, rcond=None)
    a, K = float(sol[0]), float(sol[1])
    Hfit = K / (1j * w + a)
    num = np.sum(w8 * np.abs(H - Hfit) ** 2)
    den = np.sum(w8 * np.abs(H - np.mean(H)) ** 2)
    return a, K, float(1 - num / den) if den > 0 else np.nan, Hfit


def compare_structures(fb, Hb, wb):
    """Compare candidate structures for the torque -> rate plant.

    A bare multirotor is, to first order, an INTEGRATOR (torque to rate) with an
    actuator delay; aerodynamic damping only shows up if it is large enough.
    Fitting a first-order model with a free pole forces a pole where there may be
    none — and that is how negative (non-physical) damping appears.
    """
    s_ = 1j * 2 * np.pi * fb

    def best_gain(base):
        K = np.real(np.sum(wb * np.conj(base) * Hb) / np.sum(wb * np.abs(base) ** 2))
        num = np.sum(wb * np.abs(Hb - K * base) ** 2)
        den = np.sum(wb * np.abs(Hb - np.mean(Hb)) ** 2)
        return K, float(1 - num / den)

    out = []
    best = None
    for td in np.arange(0.0, 0.201, 0.005):
        a_, K_, r_, _ = fit_first_order(fb, Hb * np.exp(1j * 2 * np.pi * fb * td), wb)
        if best is None or r_ > best[0]:
            best = (r_, a_, K_, td)
    out.append(("K e^(-s td)/(s+a), free pole",
                f"a={best[1]:+.2f} 1/s, td={1000*best[3]:.0f} ms, K={best[2]:.0f}",
                best[0]))
    best = None
    for Tm in np.arange(0.005, 0.201, 0.005):
        K, r = best_gain(1.0 / (s_ * (1 + s_ * Tm)))
        if best is None or r > best[0]:
            best = (r, Tm, K)
    out.append(("K/(s(1+s Tm)) integrator+motor",
                f"Tm={1000*best[1]:.0f} ms, K={best[2]:.0f}", best[0]))
    best = None
    for td in np.arange(0.0, 0.201, 0.005):
        K, r = best_gain(np.exp(-s_ * td) / s_)
        if best is None or r > best[0]:
            best = (r, td, K)
    out.append(("K e^(-s td)/s pure integrator",
                f"td={1000*best[1]:.0f} ms, K={best[2]:.0f}", best[0]))
    best = None
    for Tm in np.arange(0.005, 0.151, 0.005):
        for td in np.arange(0.0, 0.101, 0.005):
            K, r = best_gain(np.exp(-s_ * td) / (s_ * (1 + s_ * Tm)))
            if best is None or r > best[0]:
                best = (r, Tm, td, K)
    out.append(("K e^(-s td)/(s(1+s Tm))",
                f"Tm={1000*best[1]:.0f}, td={1000*best[2]:.0f} ms, K={best[3]:.0f}",
                best[0]))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=20.0)
    ap.add_argument("--modes", nargs="*", default=["Manual"])
    ap.add_argument("--lag", type=float, default=0.0,
                    help="lag applied to the input before the computation [s]")
    ap.add_argument("--nperseg", type=float, default=8.0, help="Welch segment [s]")
    ap.add_argument("--coh", type=float, default=0.6, help="coherence threshold")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status"])
    t, X, Xdot, U, g = sr.build_dataset(ulog, args.hz, args.fc, args.modes,
                                        None, None, "torque")
    if args.lag:
        U = sr.first_order_lag(U, args.lag, args.hz)
    name = os.path.basename(args.ulg)[:8]
    nper = int(args.nperseg * args.hz)

    print("=" * 78)
    print(f"Frequency-domain identification — {name}")
    print(f"{len(t)} samples @ {args.hz:g} Hz, modes={args.modes}, "
          f"Welch segment {args.nperseg:g} s "
          f"(resolution {1/args.nperseg:.3f} Hz), coherence threshold {args.coh}")

    axes = [("p", 0), ("q", 1), ("r", 2)]
    fig, axs = plt.subplots(3, 3, figsize=(14, 9), layout="constrained")
    fig.patch.set_facecolor(BG)
    summary = []
    for row, (nm, i) in enumerate(axes):
        x = U[:, i]                      # commanded torque
        y = X[:, i]                      # measured rate
        f, Pxx = welch(x, fs=args.hz, nperseg=nper)
        _, Pyy = welch(y, fs=args.hz, nperseg=nper)
        _, Pxy = csd(x, y, fs=args.hz, nperseg=nper)
        _, coh = coherence(x, y, fs=args.hz, nperseg=nper)
        H = Pxy / np.where(Pxx == 0, np.nan, Pxx)

        band = (f > 0.05) & (f < args.fc) & (coh >= args.coh)
        if band.sum() >= 5:
            fb, Hb, wb = f[band], H[band], coh[band]
            # Pure-delay sweep: H = K*exp(-jw*td)/(jw + a). Compensating the delay
            # before the first-order fit keeps the problem linear.
            best = None
            for td in np.arange(0.0, 0.201, 0.005):
                Hc = Hb * np.exp(1j * 2 * np.pi * fb * td)
                a_, K_, r_, _ = fit_first_order(fb, Hc, wb)
                if best is None or r_ > best[2]:
                    best = (a_, K_, r_, td)
            a, K, r2f, td = best
            fmin, fmax = fb.min(), fb.max()
            ok = True
        else:
            a = K = r2f = td = np.nan
            fmin = fmax = np.nan
            ok = False
        summary.append((nm, a, K, r2f, fmin, fmax, int(band.sum()),
                        float(coh.max()), td))

        ax = axs[row, 0]
        ax.loglog(f[1:], np.abs(H[1:]), color=C_A, lw=1.2)
        if ok:
            ax.loglog(f[band], np.abs(K / (1j * 2 * np.pi * f[band] + a)),
                      color=C_B, lw=1.4, ls="--")
        ax.set_ylabel(f"|H| {nm}", fontsize=9)
        ax.set_title(f"gain {nm}: torque -> rate", fontsize=9, loc="left")

        ax = axs[row, 1]
        ax.semilogx(f[1:], np.angle(H[1:], deg=True), color=C_A, lw=1.2)
        if ok:
            ax.semilogx(f[band],
                        np.angle(K * np.exp(-1j * 2 * np.pi * f[band] * td)
                                 / (1j * 2 * np.pi * f[band] + a), deg=True),
                        color=C_B, lw=1.4, ls="--")
        ax.set_ylabel("phase [deg]", fontsize=9)
        ax.set_title(f"phase {nm}", fontsize=9, loc="left")

        ax = axs[row, 2]
        ax.semilogx(f[1:], coh[1:], color=C_A, lw=1.2)
        ax.axhline(args.coh, color=C_B, lw=1, ls="--")
        ax.set_ylim(0, 1); ax.set_ylabel("coherence", fontsize=9)
        ax.set_title(f"coherence {nm}", fontsize=9, loc="left")

        for ax in axs[row]:
            ax.set_facecolor(BG); ax.grid(True, which="both", color=GRID, lw=0.5)
            ax.tick_params(colors=MUTED, labelsize=8)
            for s in ax.spines.values():
                s.set_color(GRID)
            if row == 2:
                ax.set_xlabel("frequency [Hz]", fontsize=9)

    print(f"\n{'axis':5} {'coherent band':>18} {'pts':>5} {'max coh':>9} "
          f"{'delay ms':>10} {'a (damping) 1/s':>16} {'K (gain)':>10} {'R2':>7}")
    for nm, a, K, r2f, fmin, fmax, n, cmax, td in summary:
        rng = f"{fmin:.2f}-{fmax:.2f} Hz" if np.isfinite(fmin) else "none"
        print(f"{nm:5} {rng:>18} {n:5} {cmax:9.2f} {1000*td:10.0f} "
              f"{a:16.3f} {K:10.1f} {r2f:7.3f}")
    print("\nPositive `a` = physical damping; negative = non-physical "
          "(closed-loop residue, or the wrong model structure).")

    print("\n--- which structure describes the plant? (R2 weighted by coherence)")
    for nm, i in axes[:2]:
        x, y = U[:, i], X[:, i]
        f, Pxx = welch(x, fs=args.hz, nperseg=nper)
        _, Pxy = csd(x, y, fs=args.hz, nperseg=nper)
        _, coh = coherence(x, y, fs=args.hz, nperseg=nper)
        H = Pxy / np.where(Pxx == 0, np.nan, Pxx)
        b = (f > 0.25) & (f < args.fc) & (coh >= args.coh)
        if b.sum() < 5:
            print(f"  {nm}: coherent band too small")
            continue
        print(f"  axis {nm}:")
        for model, par, r in compare_structures(f[b], H[b], coh[b]):
            print(f"    {model:34} {par:34} R2={r:6.3f}")

    fig.suptitle(f"Frequency response and coherence — {name}", fontsize=11,
                 color=INK, x=0.01, ha="left")
    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"freq_{name}.png")
    fig.savefig(png, dpi=130)
    print(f"figure: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

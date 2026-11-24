"""Identification with a library restricted to physically admissible terms.

Euler rigid-body model on principal axes, with the three equations coupled by a
single pair of inertia ratios ky = Iy/Ix, kz = Iz/Ix:

    p_dot = (ky - kz)     * q*r + (bx0 + bx1*T) * tau_x - cx * p
    q_dot = (kz - 1)/ky   * p*r + (by0 + by1*T) * tau_y - cy * q
    r_dot = (1 - ky)/kz   * p*q + (bz0 + bz1*T) * tau_z - cz * r

Constraints: positive control effectiveness (b >= 0), non-negative aerodynamic
damping (c >= 0) and positive inertia ratios. No constant term and no cross
terms without a physical justification.

Given (ky, kz) the problem is linear in the remaining parameters, so the fit is
a two-dimensional search with bounded least squares inside. That makes it cheap
to sweep the whole cost surface and say whether the inertias are identifiable at
all in this flight, instead of just reporting a number.

Usage:
  python src/sindy_physical.py data/ulg/<id>/<id>.ulg [--hz 50] [--modes Manual]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import sindy_rates as sr  # noqa: E402

R2D = 180.0 / np.pi
C_MEAS, C_MODEL = "#1F6FB2", "#D1670B"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e0"


def inertia_coefs(ky: float, kz: float) -> np.ndarray:
    """Gyroscopic coefficients from the inertia ratios (Ix = 1)."""
    return np.array([ky - kz, (kz - 1.0) / ky, (1.0 - ky) / kz])


def fit_linear_part(X, Xdot, U, a, idx, bounded=True):
    """With the gyroscopic coefficients fixed, fit [b0, b1, c] for one axis."""
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    T = U[:, 3]
    gyro = [q * r, p * r, p * q][idx]
    tau = U[:, idx]
    state = [p, q, r][idx]
    A = np.column_stack([tau, tau * T, -state])
    y = Xdot[:, idx] - a[idx] * gyro
    if bounded:
        res = lsq_linear(A, y, bounds=(np.zeros(3), np.full(3, np.inf)))
        beta, resid = res.x, res.fun
    else:
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = A @ beta - y
    return beta, resid, A, y


def cost(ky, kz, X, Xdot, U, mask, bounded=True):
    a = inertia_coefs(ky, kz)
    tot, parts = 0.0, []
    for i in range(3):
        beta, resid, _, _ = fit_linear_part(X[mask], Xdot[mask], U[mask], a, i, bounded)
        sse = float(np.sum(resid ** 2))
        tot += sse
        parts.append((beta, sse))
    return tot, parts


def r2(y, yhat):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else np.nan


def predict(X, U, a, betas):
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    T = U[:, 3]
    gyros = [q * r, p * r, p * q]
    states = [p, q, r]
    out = []
    for i in range(3):
        b0, b1, c = betas[i]
        out.append(a[i] * gyros[i] + (b0 + b1 * T) * U[:, i] - c * states[i])
    return np.column_stack(out)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=None)
    ap.add_argument("--modes", nargs="*", default=["Manual"])
    ap.add_argument("--lag", type=float, default=0.06)
    ap.add_argument("--input", choices=["auto", "torque", "motors"], default="torque")
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--chunks", type=int, default=6)
    args = ap.parse_args(argv)

    fc = args.fc if args.fc else min(15.0, 0.2 * args.hz)
    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status"])
    t, X, Xdot, U, g = sr.build_dataset(ulog, args.hz, fc, args.modes, None, None,
                                        args.input)
    U = sr.first_order_lag(U, args.lag, args.hz)
    tr, te = sr.block_split(t)
    name = os.path.basename(args.ulg)[:8]

    print("=" * 76)
    print(f"Physics-constrained model — {name}")
    print(f"input={g['source']}  lag={args.lag*1000:.0f} ms  grid={args.hz:g} Hz  "
          f"{len(t)} samples  modes={args.modes}")

    # ------------------------------------------------------------- fit
    def obj(z):
        return cost(np.exp(z[0]), np.exp(z[1]), X, Xdot, U, tr)[0]

    best = None
    for ky0, kz0 in [(1.0, 2.0), (1.0, 1.0), (1.2, 2.2), (0.8, 1.5), (1.0, 0.5)]:
        res = minimize(obj, np.log([ky0, kz0]), method="L-BFGS-B",
                       bounds=[(np.log(0.3), np.log(3.0)),
                               (np.log(0.3), np.log(4.0))])
        if best is None or res.fun < best.fun:
            best = res
    ky, kz = np.exp(best.x)
    sse_free = best.fun
    # physical reference point: an approximately flat, symmetric quad
    sse_planar = cost(1.0, 2.0, X, Xdot, U, tr)[0]
    # inertia-tensor triangle inequality: |Ix-Iy| <= Iz <= Ix+Iy
    tri_ok = abs(1.0 - ky) <= kz <= 1.0 + ky
    a = inertia_coefs(ky, kz)
    _, parts = cost(ky, kz, X, Xdot, U, tr)
    betas = [pp[0] for pp in parts]

    print(f"\nfitted inertia ratios (unconstrained):  Iy/Ix = {ky:.3f}   "
          f"Iz/Ix = {kz:.3f}")
    print(f"  triangle inequality (|Ix-Iy| <= Iz <= Ix+Iy): "
          f"{'satisfied' if tri_ok else 'VIOLATED'}")
    print(f"  SSE at the free optimum     = {sse_free:.6g}")
    print(f"  SSE at (Iy/Ix=1, Iz/Ix=2)   = {sse_planar:.6g}  "
          f"({100*(sse_planar-sse_free)/sse_free:+.2f}% relative to the optimum)")
    print("  -> a flat, symmetric quad has Iz ~ Ix+Iy, i.e. Iz/Ix ~ 2;"
          " Iz/Ix << 1 only exists for a body elongated along the vertical axis")
    print(f"gyroscopic coefficients:   (Iy-Iz)/Ix = {a[0]:+.3f}   "
          f"(Iz-Ix)/Iy = {a[1]:+.3f}   (Ix-Iy)/Iz = {a[2]:+.3f}")
    print("\nper-axis parameters (b0 + b1*T = effectiveness; c = damping):")
    axes_n = ["p_dot", "q_dot", "r_dot"]
    for i, nm in enumerate(axes_n):
        b0, b1, c = betas[i]
        print(f"  {nm}:  b0 = {b0:8.2f}   b1 = {b1:8.2f}   c = {c:7.3f} 1/s"
              + (f"   (tau = {1/c:.2f} s)" if c > 1e-6 else "   (no damping)"))

    Yhat = predict(X, U, a, betas)
    print("\nfit (R2):")
    for i, nm in enumerate(axes_n):
        print(f"  {nm}: train {r2(Xdot[tr, i], Yhat[tr, i]):6.3f}   "
              f"test {r2(Xdot[te, i], Yhat[te, i]):6.3f}   "
              f"residual RMS {np.std(Xdot[te, i]-Yhat[te, i])*R2D:7.1f} deg/s^2")

    # ------------------------------------- identifiability of the inertias
    print("\n--- is the gyroscopic term visible above the noise?")
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    gyros = [q * r, p * r, p * q]
    gyro_names = ["q*r", "p*r", "p*q"]
    for i, nm in enumerate(axes_n):
        contrib = np.std(a[i] * gyros[i]) * R2D
        resid = np.std(Xdot[te, i] - Yhat[te, i]) * R2D
        print(f"  {nm}: contribution of {gyro_names[i]:4} = {contrib:7.2f} deg/s^2   "
              f"residual = {resid:7.1f} deg/s^2   ratio = {contrib/resid:6.3f}")

    # cost surface (profile) over (ky, kz)
    n = args.grid
    kys = np.linspace(0.4, 2.2, n)
    kzs = np.linspace(0.4, 3.6, n)
    Z = np.empty((n, n))
    for i, kzi in enumerate(kzs):
        for j, kyj in enumerate(kys):
            Z[i, j] = cost(kyj, kzi, X, Xdot, U, tr)[0]
    Zmin = Z.min()
    rel = (Z - Zmin) / Zmin * 100.0  # percentage increase of the SSE
    inside = rel < 1.0
    print(f"\n--- cost surface over (Iy/Ix, Iz/Ix), {n}x{n} grid")
    print(f"  largest SSE increase across the swept range: {rel.max():.2f}%")
    print(f"  fraction of the range within 1% of the minimum: "
          f"{100*inside.mean():.1f}%")
    if inside.any():
        print(f"  Iy/Ix range within 1%: "
              f"{kys[inside.any(axis=0)].min():.2f} to {kys[inside.any(axis=0)].max():.2f}")
        print(f"  Iz/Ix range within 1%: "
              f"{kzs[inside.any(axis=1)].min():.2f} to {kzs[inside.any(axis=1)].max():.2f}")

    # how much excitation was missing
    print("\n--- excitation required to see the inertias")
    for i, nm in enumerate(axes_n):
        resid = np.std(Xdot[te, i] - Yhat[te, i])
        need = 3 * resid / max(abs(a[i]), 1e-6)   # |gyro| to contribute 3x the residual
        have = np.std(gyros[i])
        print(f"  {nm}: current std({gyro_names[i]}) = {have:7.4f} rad^2/s^2   "
              f"required ~{need:7.4f}   factor {need/max(have,1e-9):6.1f}x")

    # ----- decisive test: does the physical gyroscopic term improve prediction?
    print("\n--- does the gyroscopic term with flat-quad inertia "
          "(Iy/Ix=1, Iz/Ix=2) help?")
    a_ref = inertia_coefs(1.0, 2.0)
    for i, nm in enumerate(axes_n):
        b_with = fit_linear_part(X[tr], Xdot[tr], U[tr], a_ref, i)[0]
        b_none = fit_linear_part(X[tr], Xdot[tr], U[tr], np.zeros(3), i)[0]
        Bw = [b_with if j == i else np.zeros(3) for j in range(3)]
        Bn = [b_none if j == i else np.zeros(3) for j in range(3)]
        rw = r2(Xdot[te, i], predict(X, U, a_ref, Bw)[te, i])
        rn = r2(Xdot[te, i], predict(X, U, np.zeros(3), Bn)[te, i])
        verdict = "helps" if rw - rn > 0.005 else ("indifferent" if rw - rn > -0.005
                                                   else "HURTS")
        print(f"  {nm}: with term R2 = {rw:.4f}   without term R2 = {rn:.4f}   "
              f"gain = {rw-rn:+.4f}  -> {verdict}")

    # ------------------- stability: does each segment give the same number?
    print("\n--- stability of the estimate (independent fit per segment)")
    nchunk = args.chunks
    edges = np.array_split(np.arange(len(t)), nchunk)
    kys_c, kzs_c = [], []
    for ci, ids in enumerate(edges):
        m = np.zeros(len(t), dtype=bool)
        m[ids] = True

        def objc(z, m=m):
            return cost(np.exp(z[0]), np.exp(z[1]), X, Xdot, U, m)[0]

        rc = minimize(objc, np.log([1.0, 1.0]), method="L-BFGS-B",
                      bounds=[(np.log(0.3), np.log(3.0)), (np.log(0.3), np.log(4.0))])
        kyc, kzc = np.exp(rc.x)
        kys_c.append(kyc); kzs_c.append(kzc)
        print(f"  segment {ci+1} ({t[ids[0]]:6.1f}-{t[ids[-1]]:6.1f} s): "
              f"Iy/Ix = {kyc:5.2f}   Iz/Ix = {kzc:5.2f}")
    kys_c, kzs_c = np.array(kys_c), np.array(kzs_c)
    print(f"  spread: Iy/Ix = {kys_c.mean():.2f} +- {kys_c.std():.2f}   "
          f"Iz/Ix = {kzs_c.mean():.2f} +- {kzs_c.std():.2f}")
    print(f"  range: Iy/Ix {kys_c.min():.2f}..{kys_c.max():.2f}   "
          f"Iz/Ix {kzs_c.min():.2f}..{kzs_c.max():.2f}")

    # ----------------------------- comparison with the unconstrained library
    print("\n--- comparison with the unconstrained library (src/sindy_rates.py)")
    Theta, names = sr.library(X, U)
    for i, nm in enumerate(axes_n):
        xi, _ = sr.stlsq(Theta[tr], Xdot[tr, i], 0.1)
        r2u = r2(Xdot[te, i], Theta[te] @ xi)
        r2p = r2(Xdot[te, i], Yhat[te, i])
        print(f"  {nm}: physical {r2p:6.3f}  vs  unconstrained {r2u:6.3f}   "
              f"(difference {r2u-r2p:+.3f}, {int(np.sum(np.abs(xi)>0))} terms)")

    # ------------------------------------------------------------ figure
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax = axs[0]
    cs = ax.contourf(kys, kzs, np.clip(rel, 0, 5), levels=20, cmap="BuPu_r")
    ax.contour(kys, kzs, rel, levels=[0.1, 0.5, 1.0], colors=[C_MODEL], linewidths=1.0)
    ax.plot([ky], [kz], "o", color=C_MODEL, ms=7, label="fit")
    ax.plot(kys_c, kzs_c, "x", color="#444", ms=6, mew=1.4,
            label=f"per-segment fit (n={nchunk})")
    ax.plot([1.0], [2.0], "s", color=C_MEAS, ms=7, label="ideal flat quad (Iz=Ix+Iy)")
    ax.set_xlabel("Iy / Ix", color=INK, fontsize=9)
    ax.set_ylabel("Iz / Ix", color=INK, fontsize=9)
    ax.set_title("SSE increase [%] — inertia-ratio plane",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.colorbar(cs, ax=ax, label="% above the minimum")

    ax = axs[1]
    labels, contribs, resids = [], [], []
    for i, nm in enumerate(axes_n):
        labels.append(nm)
        contribs.append(np.std(a[i] * gyros[i]) * R2D)
        resids.append(np.std(Xdot[te, i] - Yhat[te, i]) * R2D)
    xpos = np.arange(3)
    ax.bar(xpos - 0.2, resids, 0.38, color=C_MEAS, label="model residual")
    ax.bar(xpos + 0.2, contribs, 0.38, color=C_MODEL, label="gyroscopic term")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, color=INK)
    ax.set_yscale("log")
    ax.set_ylabel("standard deviation [deg/s²]", color=INK, fontsize=9)
    ax.set_title("the signal that carries the inertia sits below the noise",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    for a_ in axs:
        a_.set_facecolor("#fcfcfb")
        a_.tick_params(colors=MUTED, labelsize=8)
        for s in a_.spines.values():
            s.set_color(GRID)
    axs[1].grid(True, axis="y", color=GRID, lw=0.6)
    fig.suptitle(f"Inertia from flight data — {name}", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join("out", f"inertia_{name}.png")
    fig.savefig(out, dpi=130)
    print(f"\nfigure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

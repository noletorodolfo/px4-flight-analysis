"""SINDy in weak (integral) form: identification without differentiating data.

Instead of estimating dx/dt numerically (which amplifies noise), the equation is
multiplied by a compactly supported test function phi and integrated by parts:

    integral( phi * x_dot ) dt  =  -integral( phi' * x ) dt

The left-hand side, which required the derivative, becomes an integral of the
signal itself. The right-hand side, integral( phi * Theta(x,u) ) dt, is also just
integration. No numerical derivative enters the regression.

This module repeats, in weak form, the two analyses done in the time domain:
  - a fit with the free library (compare with src/sindy_rates.py);
  - the physical Euler model with shared inertia ratios and a sweep of the cost
    surface (compare with src/sindy_physical.py).

Usage:
  python src/sindy_weak.py data/ulg/<id>/<id>.ulg [--hz 50] [--modes Manual]
                           [--window 1.0] [--lag 0.06]
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
from sindy_physical import inertia_coefs, r2  # noqa: E402

R2D = 180.0 / np.pi
C_A, C_B = "#1F6FB2", "#D1670B"
INK, MUTED, GRID, BG = "#1a1a1a", "#6b6b6b", "#e3e3e0", "#fcfcfb"


def test_function(L: int, hz: float, p: int = 6):
    """phi(tau) = (1 - tau^2)^p on [-1, 1], with phi(+-1) = phi'(+-1) = 0."""
    tau = np.linspace(-1.0, 1.0, L)
    phi = (1.0 - tau ** 2) ** p
    dphi_dtau = -2.0 * p * tau * (1.0 - tau ** 2) ** (p - 1)
    half_window_s = (L - 1) / hz / 2.0
    dphi_dt = dphi_dtau / half_window_s
    return phi, dphi_dt


def build_weak(cols: np.ndarray, target: np.ndarray, hz: float,
               window_s: float = 1.0, step_frac: float = 0.5, p: int = 6):
    """Build A (integral of phi*column) and b (-integral of phi'*target).

    cols: (n, m) regressors;  target: (n, k) states whose derivative is modelled.
    Returns A (W, m), b (W, k) and the centre index of each window.
    """
    n = cols.shape[0]
    L = max(8, int(round(window_s * hz)))
    step = max(1, int(round(L * step_frac)))
    phi, dphi = test_function(L, hz)
    norm = np.trapezoid(phi, dx=1.0 / hz)      # keeps the units of the signal
    A, b, centres = [], [], []
    for s0 in range(0, n - L + 1, step):
        sl = slice(s0, s0 + L)
        A.append(np.trapezoid(phi[:, None] * cols[sl], dx=1.0 / hz, axis=0) / norm)
        b.append(-np.trapezoid(dphi[:, None] * target[sl], dx=1.0 / hz, axis=0) / norm)
        centres.append(s0 + L // 2)
    return np.array(A), np.array(b), np.array(centres)


def library_cols(X, U):
    Theta, names = sr.library(X, U)
    return Theta, names


def physical_cols(X, U, i, a_i):
    """Euler-model columns for axis i, with the gyroscopic term already fixed."""
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    T = U[:, 3]
    gyro = [q * r, p * r, p * q][i]
    tau = U[:, i]
    state = [p, q, r][i]
    free = np.column_stack([tau, tau * T, -state])   # b0, b1, c
    fixed = a_i * gyro                                # moves to the known side
    return free, fixed


def fit_physical(X, U, hz, window_s, ky, kz, win_mask=None):
    """Fit b0,b1,c per axis in weak form, given (ky,kz). Returns SSE and betas."""
    a = inertia_coefs(ky, kz)
    sse_total, betas, preds, targets = 0.0, [], [], []
    for i in range(3):
        free, fixed = physical_cols(X, U, i, a[i])
        cols = np.column_stack([free, fixed])
        A, b, _ = build_weak(cols, X[:, [i]], hz, window_s)
        if win_mask is not None:
            A, b = A[win_mask], b[win_mask]
        y = b[:, 0] - A[:, 3]            # move the gyroscopic term to the target
        M = A[:, :3]
        res = lsq_linear(M, y, bounds=(np.zeros(3), np.full(3, np.inf)))
        betas.append(res.x)
        sse_total += float(np.sum(res.fun ** 2))
        preds.append(M @ res.x)
        targets.append(y)
    return sse_total, betas, preds, targets


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=None)
    ap.add_argument("--modes", nargs="*", default=["Manual"])
    ap.add_argument("--window", type=float, default=1.0,
                    help="width of the test function [s]")
    ap.add_argument("--lag", type=float, default=0.06)
    ap.add_argument("--input", choices=["auto", "torque", "motors"], default="torque")
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--chunks", type=int, default=8)
    args = ap.parse_args(argv)

    fc = args.fc if args.fc else min(15.0, 0.2 * args.hz)
    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status"])
    t, X, Xdot, U, g = sr.build_dataset(ulog, args.hz, fc, args.modes, None, None,
                                        args.input)
    U = sr.first_order_lag(U, args.lag, args.hz)
    name = os.path.basename(args.ulg)[:8]

    print("=" * 78)
    print(f"SINDy in WEAK FORM — {name}")
    print(f"input={g['source']}  lag={args.lag*1000:.0f} ms  grid={args.hz:g} Hz  "
          f"low-pass={fc:g} Hz  test window={args.window:g} s  "
          f"{len(t)} samples")

    # ----------------------------------------------- free library, weak form
    Theta, names = library_cols(X, U)
    A, B, centres = build_weak(Theta, X, args.hz, args.window)
    tj = t[centres]
    tr_j, te_j = sr.block_split(tj, 10.0)
    print(f"test windows: {len(A)} ({tr_j.sum()} train / {te_j.sum()} test)")

    print("\n--- free library: weak form vs strong form (derivative)")
    for k, axis in enumerate(["p_dot", "q_dot", "r_dot"]):
        # weak
        best = None
        for th in [0.002, 0.01, 0.05, 0.1, 0.2, 0.35]:
            xi, _ = sr.stlsq(A[tr_j], B[tr_j, k], th)
            sc = r2(B[te_j, k], A[te_j] @ xi)
            n_t = int(np.sum(np.abs(xi) > 0))
            if best is None or sc > best[0]:
                best = (sc, th, xi, n_t)
        sc_w, th_w, xi_w, nt_w = best
        # strong form, same window selection
        tr_s, te_s = sr.block_split(t, 10.0)
        best_s = None
        for th in [0.002, 0.01, 0.05, 0.1, 0.2, 0.35]:
            xi, _ = sr.stlsq(Theta[tr_s], Xdot[tr_s, k], th)
            sc = r2(Xdot[te_s, k], Theta[te_s] @ xi)
            if best_s is None or sc > best_s[0]:
                best_s = (sc, th, xi, int(np.sum(np.abs(xi) > 0)))
        sc_s, _, xi_s, nt_s = best_s
        # the weak model evaluated against the derivative, for comparability
        sc_cross = r2(Xdot[te_s, k], Theta[te_s] @ xi_w)
        print(f"  {axis}: weak R2 {sc_w:6.3f} ({nt_w} terms) | "
              f"strong R2 {sc_s:6.3f} ({nt_s} terms) | "
              f"weak model scored on the derivative: {sc_cross:6.3f}")

    # ------------------------------------- physical model, in weak form
    print("\n--- physical Euler model, now in weak form")

    def obj(z):
        return fit_physical(X, U, args.hz, args.window, np.exp(z[0]), np.exp(z[1]))[0]

    best = None
    for ky0, kz0 in [(1.0, 2.0), (1.0, 1.0), (1.2, 2.2), (0.8, 1.5), (1.0, 0.5)]:
        res = minimize(obj, np.log([ky0, kz0]), method="L-BFGS-B",
                       bounds=[(np.log(0.3), np.log(3.0)), (np.log(0.3), np.log(4.0))])
        if best is None or res.fun < best.fun:
            best = res
    ky, kz = np.exp(best.x)
    a = inertia_coefs(ky, kz)
    sse_free, betas, preds, targets = fit_physical(X, U, args.hz, args.window, ky, kz)
    sse_flat = fit_physical(X, U, args.hz, args.window, 1.0, 2.0)[0]
    print(f"  fitted ratios: Iy/Ix = {ky:.3f}   Iz/Ix = {kz:.3f}")
    print(f"  coefficients: (Iy-Iz)/Ix={a[0]:+.3f}  (Iz-Ix)/Iy={a[1]:+.3f}  "
          f"(Ix-Iy)/Iz={a[2]:+.3f}")
    print(f"  SSE at the free optimum = {sse_free:.6g};  at (1, 2) = {sse_flat:.6g} "
          f"({100*(sse_flat-sse_free)/sse_free:+.2f}%)")
    for i, nm in enumerate(["p_dot", "q_dot", "r_dot"]):
        b0, b1, c = betas[i]
        print(f"    {nm}: b0={b0:8.2f}  b1={b1:8.2f}  c={c:7.3f} 1/s   "
              f"weak R2={r2(targets[i], preds[i]):6.3f}")

    # decisive test, in weak form
    print("\n--- does the physical gyroscopic term (Iy/Ix=1, Iz/Ix=2) help in weak "
          "form?")
    a_ref = inertia_coefs(1.0, 2.0)
    for i, nm in enumerate(["p_dot", "q_dot", "r_dot"]):
        _, _, pr_c, al_c = fit_physical(X, U, args.hz, args.window, 1.0, 2.0)
        _, _, pr_s, al_s = fit_physical(X, U, args.hz, args.window, 1.0, 1.0)  # a=0 em p_dot
        rc = r2(al_c[i], pr_c[i])
        # model without the gyroscopic term: a = 0 on every axis
        free, _ = physical_cols(X, U, i, 0.0)
        A0, b0_, _ = build_weak(np.column_stack([free, np.zeros(len(X))]),
                                X[:, [i]], args.hz, args.window)
        res0 = lsq_linear(A0[:, :3], b0_[:, 0], bounds=(np.zeros(3), np.full(3, np.inf)))
        rn = r2(b0_[:, 0], A0[:, :3] @ res0.x)
        verdict = ("helps" if rc - rn > 0.005 else
                   "indifferent" if rc - rn > -0.005 else "HURTS")
        print(f"  {nm}: with term R2 = {rc:.4f}   without term R2 = {rn:.4f}   "
              f"gain = {rc-rn:+.4f}  -> {verdict}")

    # per-segment stability
    print("\n--- per-segment stability (weak form)")
    nj = len(A)
    blocks = np.array_split(np.arange(nj), args.chunks)
    kys, kzs = [], []
    for ci, ids in enumerate(blocks):
        m = np.zeros(nj, dtype=bool); m[ids] = True

        def objc(z, m=m):
            return fit_physical(X, U, args.hz, args.window,
                                 np.exp(z[0]), np.exp(z[1]), win_mask=m)[0]

        rc = minimize(objc, np.log([1.0, 1.0]), method="L-BFGS-B",
                      bounds=[(np.log(0.3), np.log(3.0)), (np.log(0.3), np.log(4.0))])
        k1, k2 = np.exp(rc.x)
        kys.append(k1); kzs.append(k2)
        print(f"  segment {ci+1} ({tj[ids[0]]:6.1f}-{tj[ids[-1]]:6.1f} s): "
              f"Iy/Ix={k1:5.2f}  Iz/Ix={k2:5.2f}")
    kys, kzs = np.array(kys), np.array(kzs)
    print(f"  spread: Iy/Ix = {kys.mean():.2f} +- {kys.std():.2f}   "
          f"Iz/Ix = {kzs.mean():.2f} +- {kzs.std():.2f}")

    # cost surface
    n = args.grid
    gky = np.linspace(0.4, 2.2, n); gkz = np.linspace(0.4, 3.6, n)
    Z = np.empty((n, n))
    for i, kzi in enumerate(gkz):
        for j, kyj in enumerate(gky):
            Z[i, j] = fit_physical(X, U, args.hz, args.window, kyj, kzi)[0]
    rel = (Z - Z.min()) / Z.min() * 100
    print(f"\n--- cost surface (weak): largest increase {rel.max():.2f}%, "
          f"{100*np.mean(rel < 1.0):.1f}% of the range within 1% of the minimum")

    fig, ax = plt.subplots(figsize=(7.2, 5.4), layout="constrained")
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    cs = ax.contourf(gky, gkz, np.clip(rel, 0, 5), levels=20, cmap="BuPu_r")
    ax.contour(gky, gkz, rel, levels=[0.1, 0.5, 1.0], colors=[C_B], linewidths=1.0)
    ax.plot(kys, kzs, "x", color="#444", ms=6, mew=1.4, label="per-segment fit")
    ax.plot([ky], [kz], "o", color=C_B, ms=7, label="fit (weak form)")
    ax.plot([1.0], [2.0], "s", color=C_A, ms=7, label="flat quad (Iz=Ix+Iy)")
    ax.set_xlabel("Iy / Ix", fontsize=9); ax.set_ylabel("Iz / Ix", fontsize=9)
    ax.set_title("weak form — SSE increase [%] in the inertia plane",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.colorbar(cs, ax=ax, label="% above the minimum")
    out = os.path.join("out", f"inertia_weak_{name}.png")
    fig.savefig(out, dpi=130)
    print(f"figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

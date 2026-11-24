"""Instrumental-variable identification (2SLS) for closed-loop flight data.

Problem: in a normal flight the command is produced by the controller from the
state (u = -K x). Regressors and residual become correlated and the coefficients
are biased — that is how POSITIVE (non-physical) damping showed up in ordinary
least squares.

Classic remedy: use an **exogenous** signal as an instrument. Here the instrument
is the pilot command (`manual_control_setpoint`), which enters the system from
outside and is not computed from the estimated state. Two-stage least squares:

  stage 1: project each regressor onto the space spanned by the instruments Z;
  stage 2: regress the derivative on the projected regressors.

The report prints the **instrument strength** (first-stage R2): with a weak
instrument the 2SLS estimate is worse than plain least squares, and that has to
be visible rather than hidden.

Assumption (and it is debatable): the pilot reacts to the aircraft with a delay
longer than the time scale of the rate loop, so the stick is approximately
exogenous.

Usage: python src/sindy_iv.py data/ulg/<id>/<id>.ulg [--modes Manual]
                              [--lags 0 0.1 0.2 0.3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import sindy_rates as sr  # noqa: E402
from sindy_physical import inertia_coefs, r2  # noqa: E402

R2D = 180.0 / np.pi


def instruments(ulog, t_grid, hz, lags_s):
    """Matrix Z: lagged stick channels plus a constant."""
    mc = px4io.to_df(ulog, "manual_control_setpoint")
    channels = [c for c in ("roll", "pitch", "yaw", "throttle") if c in mc.columns]
    if not channels:
        return None, []
    base = np.column_stack([np.interp(t_grid, mc.index.to_numpy(),
                                      mc[c].to_numpy()) for c in channels])
    cols, names = [np.ones(len(t_grid))], ["1"]
    for L in lags_s:
        k = int(round(L * hz))
        for j, c in enumerate(channels):
            x = base[:, j]
            if k > 0:
                x = np.concatenate([np.full(k, x[0]), x[:-k]])
            cols.append(x)
            names.append(f"{c}(-{L:g}s)")
    return np.column_stack(cols), names


def two_stage(Xreg, y, Z):
    """2SLS. Returns beta, the first-stage R2 per regressor and the final R2."""
    Zq, _ = np.linalg.qr(Z)                      # orthonormal basis of span(Z)
    Xhat = Zq @ (Zq.T @ Xreg)
    r2_1 = []
    for j in range(Xreg.shape[1]):
        v = Xreg[:, j] - Xreg[:, j].mean()
        ss = float(np.sum(v ** 2))
        r2_1.append(float(1 - np.sum((Xreg[:, j] - Xhat[:, j]) ** 2) / ss)
                    if ss > 0 else np.nan)
    beta = np.linalg.lstsq(Xhat, y, rcond=None)[0]
    return beta, np.array(r2_1), r2(y, Xreg @ beta)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=10.0)
    ap.add_argument("--modes", nargs="*", default=["Manual"])
    ap.add_argument("--lag", type=float, default=0.06)
    ap.add_argument("--lags", type=float, nargs="*",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--ky", type=float, default=1.0)
    ap.add_argument("--kz", type=float, default=2.0)
    args = ap.parse_args(argv)

    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status", "manual_control_setpoint"])
    t, X, Xdot, U, g = sr.build_dataset(ulog, args.hz, args.fc, args.modes,
                                        None, None, "torque")
    U = sr.first_order_lag(U, args.lag, args.hz)
    name = os.path.basename(args.ulg)[:8]

    Z, znames = instruments(ulog, t, args.hz, args.lags)
    print("=" * 78)
    print(f"Instrumental variables (2SLS) — {name}")
    print(f"{len(t)} samples @ {args.hz:g} Hz, modes={args.modes}")
    if Z is None:
        print("no usable manual_control_setpoint: 2SLS not possible on this log")
        return 1
    print(f"instruments: {len(znames)} columns "
          f"({', '.join(znames[1:5])}, ... lags {args.lags})")

    a = inertia_coefs(args.ky, args.kz)
    p, q, r_ = X[:, 0], X[:, 1], X[:, 2]
    T = U[:, 3]
    gyros = [q * r_, p * r_, p * q]
    states = [p, q, r_]
    axes = ["p_dot", "q_dot", "r_dot"]

    print("\nper-axis model:  x_dot = a*gyro + b0*tau + b1*tau*T - c*state")
    print(f"(gyroscopic coefficients fixed at Iy/Ix={args.ky}, Iz/Ix={args.kz}: "
          f"{a.round(3)})\n")

    for i, nm in enumerate(axes):
        y = Xdot[:, i] - a[i] * gyros[i]
        Xreg = np.column_stack([U[:, i], U[:, i] * T, -states[i]])
        names = ["b0 (tau)", "b1 (tau*T)", "c (damping)"]
        ols = np.linalg.lstsq(Xreg, y, rcond=None)[0]
        r2_ols = r2(y, Xreg @ ols)
        iv, r2_first, r2_iv = two_stage(Xreg, y, Z)
        print(f"--- {nm}")
        print(f"    {'coefficient':22} {'OLS':>12} {'2SLS':>12}   "
              f"{'instrument strength (1st-stage R2)':>12}")
        for j, cn in enumerate(names):
            print(f"    {cn:22} {ols[j]:12.4g} {iv[j]:12.4g}   {r2_first[j]:12.3f}")
        print(f"    fit R2: OLS {r2_ols:.3f} | 2SLS {r2_iv:.3f}")
        c_ols, c_iv = ols[2], iv[2]
        def reading(c):
            if c > 1e-6:
                return f"damping {c:+.3g} 1/s (physical: >0 in this convention)"
            return f"damping {c:+.3g} 1/s  <- NEGATIVE, non-physical"
        print(f"    OLS : {reading(c_ols)}")
        print(f"    2SLS: {reading(c_iv)}")
        # b0 and b1 are nearly collinear (tau and tau*T): what is identifiable is
        # the effective gain at typical thrust, not the split between the terms
        Tm = float(np.mean(T))
        ge_ols = ols[0] + ols[1] * Tm
        ge_iv = iv[0] + iv[1] * Tm
        print(f"    effective control gain at mean T ({Tm:.2f}): "
              f"OLS {ge_ols:.1f}  |  2SLS {ge_iv:.1f} rad/s² per unit tau "
              f"(difference {100*(ge_iv-ge_ols)/abs(ge_ols):+.0f}%)")
        print()

    # is the instrument strong enough?
    print("--- instrument diagnostics")
    Xall = np.column_stack([U[:, 0], U[:, 1], U[:, 2], T])
    _, r2_1, _ = two_stage(Xall, Xdot[:, 0], Z)
    for nm, v in zip(["tau_x", "tau_y", "tau_z", "T"], r2_1):
        mark = ("strong" if v > 0.5 else "weak" if v > 0.2 else "VERY WEAK")
        print(f"    {nm:6}: first-stage R2 = {v:.3f}  ({mark})")
    print("\n    a weak instrument means the 2SLS estimate has high variance and "
          "may be worse than OLS; in that case the result does not remove the "
          "bias, it only trades it for noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

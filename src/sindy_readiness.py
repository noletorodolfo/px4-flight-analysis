"""Check whether a log can support dynamics identification (SINDy) on the
attitude axes.

Criteria:
  1. effective sampling rate of the required signals (p, q, r and inputs);
  2. signal bandwidth (f95) vs the Nyquist of the logged rate -> aliasing risk;
  3. excitation: standard deviation and fraction of time manoeuvring;
  4. noise in the numerical derivative (SINDy needs dx/dt);
  5. conditioning of a candidate polynomial library.

Usage: python src/sindy_readiness.py data/ulg/<id>/<id>.ulg [--hz 100]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402

R2D = 180.0 / np.pi


def f95(x: np.ndarray, fs: float) -> float:
    """Frequency below which 95% of the energy sits (DC component removed)."""
    x = x - np.mean(x)
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    P[0] = 0.0
    c = np.cumsum(P)
    if c[-1] == 0:
        return 0.0
    return float(f[np.searchsorted(c, 0.95 * c[-1])])


def effective_hz(ulog, topic, multi_id=0):
    d = ulog.get_dataset(topic, multi_id)
    t = d.data["timestamp"].astype(np.float64) * 1e-6
    dt = np.diff(t)
    dt = dt[dt > 0]
    return (len(t), float(np.median(dt)) and 1 / float(np.median(dt)),
            float(np.mean(dt)) and 1 / float(np.mean(dt)))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=100.0, help="resampling grid")
    args = ap.parse_args(argv)

    topics = ["vehicle_angular_velocity", "vehicle_attitude", "vehicle_attitude_setpoint",
              "vehicle_rates_setpoint", "vehicle_status", "actuator_motors",
              "actuator_outputs", "manual_control_setpoint"]
    ulog = px4io.load(args.ulg, topics)
    present = {d.name for d in ulog.data_list}
    name = os.path.basename(args.ulg)[:8]

    print("=" * 74)
    print(f"SINDy readiness — {name}  ({ulog.msg_info_dict.get('ver_hw','?')})")

    print("\n1) logged rates (Hz: median / mean)")
    for t in ["vehicle_angular_velocity", "vehicle_attitude", "actuator_motors",
              "actuator_outputs", "vehicle_rates_setpoint"]:
        if t in present:
            n, hm, ha = effective_hz(ulog, t)
            print(f"   {t:28} n={n:7}  {hm:7.1f} / {ha:7.1f}")
        else:
            print(f"   {t:28} MISSING")

    rate = px4io.to_df(ulog, "vehicle_angular_velocity")[["xyz[0]", "xyz[1]", "xyz[2]"]]
    rate.columns = ["p", "q", "r"]
    att = px4io.euler_df(ulog, "vehicle_attitude", "a")
    frames = [rate, att]
    u_topic = "actuator_motors" if "actuator_motors" in present else "actuator_outputs"
    u = px4io.to_df(ulog, u_topic)
    ucols = [c for c in u.columns if c.startswith(("control[", "output["))][:8]
    if ucols:
        u = u[ucols]
        # actuator_outputs comes in PWM (us); roughly normalise to [0,1]
        if u_topic == "actuator_outputs" and np.nanmax(u.to_numpy()) > 10:
            u = (u - 1000.0) / 1000.0
        frames.append(u)
    df = px4io.resample_join(frames, args.hz)

    armed = px4io.armed_intervals(ulog)
    if armed:
        t0, t1 = armed[0][0], armed[-1][1]
        df = df.loc[(df.index >= t0) & (df.index <= t1)]
    fs = args.hz
    p, q, r = (df[c].to_numpy() for c in ("p", "q", "r"))

    print(f"\n2) signal bandwidth (f95) vs Nyquist (armed window, {len(df)} "
          f"samples @ {fs:g} Hz)")
    nyq_log = effective_hz(ulog, "vehicle_angular_velocity")[1] / 2
    for lbl, x in (("p", p), ("q", q), ("r", r)):
        f = f95(x, fs)
        print(f"   {lbl}: f95 = {f:6.2f} Hz   (log Nyquist = {nyq_log:.0f} Hz)"
              f"{'   <- aliasing risk' if f > 0.8 * min(nyq_log, fs/2) else ''}")

    print("\n3) excitation")
    for lbl, x in (("p", p), ("q", q), ("r", r)):
        xd = x * R2D
        frac = float(np.mean(np.abs(xd) > 20))
        print(f"   {lbl}: std = {np.std(xd):7.2f} deg/s   max = {np.max(np.abs(xd)):7.1f}"
              f"   time with |{lbl}|>20 deg/s = {100*frac:5.1f}%")
    for lbl in ("a_roll", "a_pitch"):
        x = df[lbl].to_numpy() * R2D
        print(f"   {lbl}: std = {np.std(x):7.2f} deg   max|.| = {np.max(np.abs(x)):7.1f} deg")

    print("\n4) derivative noise (dp/dt by central differences)")
    for lbl, x in (("p", p), ("q", q), ("r", r)):
        d1 = np.gradient(x, 1 / fs)
        # share of the derivative energy that sits above 20 Hz
        n = len(d1)
        f = np.fft.rfftfreq(n, 1 / fs)
        P = np.abs(np.fft.rfft((d1 - d1.mean()) * np.hanning(n))) ** 2
        hf = float(P[f > 20].sum() / P.sum()) if P.sum() > 0 else np.nan
        print(f"   d{lbl}/dt: std = {np.std(d1)*R2D:8.1f} deg/s^2   "
              f"energy >20 Hz = {100*hf:5.1f}%")

    print("\n5) conditioning of the candidate library (degree-2 polynomials in "
          "p, q, r, u)")
    base, names = [p, q, r], ["p", "q", "r"]
    dropped = []
    for c in ucols:
        x = df[c].to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            dropped.append(f"{c} (NaN/inf)")
            continue
        if np.std(x) < 1e-6:
            dropped.append(f"{c} (constant)")
            continue
        base.append(x)
        names.append(c)
    if dropped:
        print("   inputs discarded:", ", ".join(dropped))
    print(f"   inputs used: {', '.join(names[3:]) or '(none)'}")
    cols, cnames = [np.ones_like(p)], ["1"]
    for i, a in enumerate(base):
        cols.append(a); cnames.append(names[i])
    for i in range(len(base)):
        for j in range(i, len(base)):
            cols.append(base[i] * base[j]); cnames.append(f"{names[i]}*{names[j]}")
    Theta = np.column_stack(cols)
    keep = np.all(np.isfinite(Theta), axis=0)
    Theta, cnames = Theta[:, keep], [n for n, k in zip(cnames, keep) if k]
    scale = np.linalg.norm(Theta, axis=0)
    scale[scale == 0] = 1
    Ts = Theta / scale
    s = np.linalg.svd(Ts, compute_uv=False, full_matrices=False)
    cond = s[0] / s[-1] if s[-1] > 0 else np.inf
    rank = int(np.sum(s > s[0] * 1e-10))
    print(f"   terms = {Theta.shape[1]}   numerical rank = {rank}   "
          f"cond(normalised Theta) = {cond:.3e}")
    print(f"   {'OK (<1e3)' if cond < 1e3 else 'high: nearly collinear terms, expect '
          'unstable coefficients'}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Attitude and rates: commanded (setpoint) vs measured, from a .ulg log.

Usage: python src/plot_attitude.py data/ulg/<id>/<id>.ulg [--t0 S] [--t1 S] [--hz 50]
Writes a PNG to out/ and prints the RMS error per axis.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402

C_SP = "#D1670B"   # commanded setpoint
C_ACT = "#1F6FB2"  # measured / estimated
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e3e3e0"
R2D = 180.0 / np.pi


def build(ulog, hz: float):
    att = px4io.euler_df(ulog, "vehicle_attitude", "act", qname="q")
    sp = px4io.euler_df(ulog, "vehicle_attitude_setpoint", "sp", qname="q_d")
    rate = px4io.to_df(ulog, "vehicle_angular_velocity")[["xyz[0]", "xyz[1]", "xyz[2]"]]
    rate.columns = ["act_p", "act_q", "act_r"]
    rsp = px4io.to_df(ulog, "vehicle_rates_setpoint")[["roll", "pitch", "yaw"]]
    rsp.columns = ["sp_p", "sp_q", "sp_r"]
    return px4io.resample_join([att, sp, rate, rsp], hz)


def rms_err(df, sp, act, angle=False):
    e = df[sp].to_numpy() - df[act].to_numpy()
    if angle:
        e = px4io.wrap_pi(e)
    return float(np.sqrt(np.mean(e**2)))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    ulog = px4io.load(ulog_path := args.ulg, [
        "vehicle_attitude", "vehicle_attitude_setpoint", "vehicle_angular_velocity",
        "vehicle_rates_setpoint", "vehicle_status"])
    df = build(ulog, args.hz)
    modes = px4io.mode_intervals(ulog)
    armed = px4io.armed_intervals(ulog)

    # window: the armed period by default
    t0 = args.t0 if args.t0 is not None else (armed[0][0] if armed else df.index[0])
    t1 = args.t1 if args.t1 is not None else (armed[-1][1] if armed else df.index[-1])
    df = df.loc[(df.index >= t0) & (df.index <= t1)]

    panels = [
        ("roll [deg]", "sp_roll", "act_roll", R2D, True),
        ("pitch [deg]", "sp_pitch", "act_pitch", R2D, True),
        ("yaw [deg]", "sp_yaw", "act_yaw", R2D, True),
        ("p (roll rate) [deg/s]", "sp_p", "act_p", R2D, False),
        ("q (pitch rate) [deg/s]", "sp_q", "act_q", R2D, False),
        ("r (yaw rate) [deg/s]", "sp_r", "act_r", R2D, False),
    ]

    errs = {lbl: rms_err(df, sp, act, ang) * k for lbl, sp, act, k, ang in panels}

    # unwrap yaw for display (avoids the jumps at +-180 deg)
    for c in ("act_yaw", "sp_yaw"):
        df[c] = np.unwrap(df[c].to_numpy())

    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (lbl, sp, act, k, ang) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ax.plot(df.index, df[act] * k, color=C_ACT, lw=1.2, label="measured")
        ax.plot(df.index, df[sp] * k, color=C_SP, lw=1.2, ls="--", label="commanded")
        ax.set_ylabel(lbl, fontsize=9, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.text(0.995, 0.04, f"RMS {errs[lbl]:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=MUTED)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    # flight-mode boundaries
    for ax in axes:
        for a, b, name in modes:
            if b < t0 or a > t1:
                continue
            ax.axvline(max(a, t0), color=MUTED, lw=0.5, alpha=0.4)
    last_name, last_x = None, -1e9
    for a, b, name in modes:
        if b < t0 or a > t1 or (b - a) < (t1 - t0) * 0.03:
            continue
        x = max(a, t0)
        if name == last_name or x - last_x < (t1 - t0) * 0.05:
            continue
        axes[0].text(x + 1, axes[0].get_ylim()[1], name, fontsize=7,
                     color=MUTED, va="top")
        last_name, last_x = name, x

    axes[-1].set_xlabel("time [s since log start]", fontsize=9, color=INK)
    name = os.path.basename(ulog_path).replace(".ulg", "")
    fig.suptitle(f"Attitude tracking — {name[:8]}  "
                 f"({ulog.msg_info_dict.get('ver_hw','?')}, window {t0:.0f}–{t1:.0f} s)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"attitude_{name[:8]}.png")
    fig.savefig(png, dpi=130)
    print(f"figure: {png}")
    print(f"window: {t0:.1f}–{t1:.1f} s @ {args.hz:g} Hz ({len(df)} samples)")
    print("RMS error (commanded - measured):")
    for lbl, v in errs.items():
        print(f"  {lbl:26} {v:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Scan .ulg logs and rank them by gyroscopic excitation.

To identify inertia from flight data, what has to show up are the Euler products
q*r, p*r and p*q. This script measures how much of them each log contains, to
answer: "is there any public flight with enough excitation?"

Reference: on log 22e75c8c, std(q*r) would have to reach ~15 rad^2/s^2 for the
gyroscopic term to sit 3x above the model residual.

Usage: python src/excitation_scan.py data/ulg/*/*.ulg [--target 15]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402

R2D = 180.0 / np.pi


def scan(path: str, mode_filter: list[str] | None = None) -> dict | None:
    try:
        u = px4io.load(path, ["vehicle_angular_velocity", "vehicle_status",
                              "actuator_motors"])
        d = px4io.to_df(u, "vehicle_angular_velocity")
    except Exception as e:                                    # truncated log, etc.
        return {"id": os.path.basename(path)[:8], "error": str(e)[:40]}
    p = d["xyz[0]"].to_numpy(); q = d["xyz[1]"].to_numpy(); r = d["xyz[2]"].to_numpy()
    t = d.index.to_numpy()

    armed = px4io.armed_intervals(u)
    m = np.zeros(len(t), dtype=bool)
    for a, b in armed:
        m |= (t >= a) & (t <= b)
    if not m.any():
        m[:] = True
    modes = px4io.mode_intervals(u)
    if mode_filter:
        mm = np.zeros(len(t), dtype=bool)
        for a, b, nm in modes:
            if nm in mode_filter:
                mm |= (t >= a) & (t <= b)
        if mm.any():
            m &= mm
    p, q, r, t = p[m], q[m], r[m], t[m]
    if len(p) < 100:
        return None

    # Allocator saturation: with a motor on the floor or ceiling the commanded
    # torque is not delivered, so the model input becomes fiction.
    sat = np.nan
    try:
        am = px4io.to_df(u, "actuator_motors")
        cs = [c for c in am.columns if c.startswith("control[")][:8]
        ta = am.index.to_numpy()
        ma = np.zeros(len(ta), dtype=bool)
        for a, b in armed:
            ma |= (ta >= a) & (ta <= b)
        if mode_filter:
            mb = np.zeros(len(ta), dtype=bool)
            for a, b, nm in modes:
                if nm in mode_filter:
                    mb |= (ta >= a) & (ta <= b)
            if mb.any():
                ma &= mb
        U = np.column_stack([am[c].to_numpy() for c in cs])[ma]
        U = U[:, ~np.all(np.isnan(U), axis=0)]
        if U.size:
            sat = 100 * float(np.mean(np.any((U <= 0.02) | (U >= 0.95), axis=1)))
    except Exception:
        pass

    qr, pr, pq = q * r, p * r, p * q
    both = np.mean((np.abs(q) > 1.75) & (np.abs(r) > 1.75))   # ~100 deg/s on both
    return {
        "id": os.path.basename(path)[:8],
        "dur": float(t[-1] - t[0]),
        "hw": u.msg_info_dict.get("ver_hw", "?")[:16],
        "std_p": float(np.std(p)) * R2D, "std_q": float(np.std(q)) * R2D,
        "std_r": float(np.std(r)) * R2D,
        "max_p": float(np.max(np.abs(p))) * R2D,
        "max_r": float(np.max(np.abs(r))) * R2D,
        "qr": float(np.std(qr)), "pr": float(np.std(pr)), "pq": float(np.std(pq)),
        "qr_max": float(np.max(np.abs(qr))),
        "coupling": 100 * float(both), "sat": sat,
        "modes": sorted({nm for _, _, nm in modes}),
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulgs", nargs="+")
    ap.add_argument("--target", type=float, default=15.0,
                    help="required std(q*r) [rad^2/s^2]")
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--plot", default=None,
                    help="save the excitation vs saturation scatter here")
    args = ap.parse_args(argv)

    paths = []
    for g in args.ulgs:
        paths.extend(sorted(glob.glob(g)) if any(c in g for c in "*?[") else [g])

    rows = []
    for pth in paths:
        rec = scan(pth, args.modes)
        if rec is None:
            continue
        if "error" in rec:
            print(f"  {rec['id']}: could not read ({rec['error']})")
            continue
        rows.append(rec)
    rows.sort(key=lambda r: -r["qr"])

    print(f"\ntarget: std(q*r) >= {args.target:g} rad^2/s^2"
          + (f"   (modes {args.modes} only)" if args.modes else "   (whole armed flight)"))
    print("-" * 104)
    print(f"{'log':9} {'hw':16} {'dur':>6} {'std p':>7} {'std r':>7} "
          f"{'std qr':>7} {'std pr':>7} {'std pq':>7} {'max qr':>7} {'coupl%':>7} {'sat%':>6} {'target':>7}")
    print("-" * 104)
    for r in rows:
        frac = r["qr"] / args.target
        print(f"{r['id']:9} {r['hw']:16} {r['dur']:6.0f} {r['std_p']:7.1f} {r['std_r']:7.1f} "
              f"{r['qr']:7.3f} {r['pr']:7.3f} {r['pq']:7.3f} {r['qr_max']:7.2f} "
              f"{r['coupling']:7.2f} {r['sat']:6.1f} {frac:6.2f}x")
    print("-" * 104)
    print("std p/r in deg/s; products in rad^2/s^2; coupl% = time with |q| and "
          "|r| both above 100 deg/s;")
    print("sat% = time with some motor on the floor (<=0.02) or ceiling (>=0.95): "
          "under saturation the commanded torque is not delivered")
    usable = [r for r in rows if r["qr"] >= 0.3 * args.target and r["sat"] < 30]
    print(f"\nlogs in the useful corner (excitation >= 30% of target AND "
          f"saturation < 30%): {len(usable)}")
    if args.plot:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
        xs = [max(r["qr"], 1e-3) for r in rows]
        ys = [r["sat"] if np.isfinite(r["sat"]) else -3 for r in rows]
        ax.axvspan(0.3 * args.target, 1e3, color="#1F6FB2", alpha=0.06)
        ax.axhspan(-5, 30, color="#1F6FB2", alpha=0.06)
        ax.scatter(xs, ys, s=38, color="#1F6FB2", zorder=3)
        for r, x, y in zip(rows, xs, ys):
            if r["qr"] > 0.2 or r["sat"] > 50:
                ax.annotate(r["id"], (x, y), fontsize=7, color="#6b6b6b",
                            xytext=(4, 4), textcoords="offset points")
        ax.axvline(args.target, color="#D1670B", lw=1.2)
        ax.annotate(f"target {args.target:g}", (args.target, 92), color="#D1670B",
                    fontsize=8, rotation=90, va="top", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel("gyroscopic excitation — std(q·r) [rad²/s²]", fontsize=9)
        ax.set_ylabel("allocator saturation [% of armed time]", fontsize=9)
        ax.set_title("no public log sits in the useful corner "
                     "(high excitation AND low saturation)", fontsize=10, loc="left")
        ax.grid(True, color="#e3e3e0", lw=0.6)
        ax.tick_params(colors="#6b6b6b", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#e3e3e0")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"figure: {args.plot}")

    best = rows[0] if rows else None
    if best:
        print(f"\nbest: {best['id']} with std(q*r) = {best['qr']:.3f} "
              f"({best['qr']/args.target:.2f}x the target) — modes: "
              f"{', '.join(best['modes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

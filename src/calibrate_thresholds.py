"""Calibrate the report thresholds against a sample of the public fleet.

The thresholds in `src/flight_report.py` started as my own guesses. This script
downloads a sample of public logs (one per vehicle, so the distribution is not
dominated by one operator), measures them with the same functions the report
uses, deletes each log immediately — disk usage stays flat — and prints the
empirical distribution of every metric.

A threshold then stops being a guess and becomes a percentile: "worse than 90 %
of the public fleet" is a checkable statement.

Usage:
  python src/calibrate_thresholds.py ids.txt [--max 220] [--out out/calibration.json]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import flight_report as fr  # noqa: E402

URL = "https://cdn.logs.px4.io/{}.ulg"


def download(log_id: str, dest: str, limit_bytes: int = 60_000_000) -> str | None:
    target = os.path.join(dest, f"{log_id}.ulg")
    r = subprocess.run(["curl", "-sSL", "-m", "240", "--max-filesize",
                        str(limit_bytes), "-o", target, URL.format(log_id)],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(target) or os.path.getsize(target) < 10000:
        return None
    return target


def measure(path: str) -> dict | None:
    """Measure with the SAME functions the report uses, so the calibration
    cannot drift away from what is being thresholded."""
    u = px4io.load(path)
    ctx = {"u": u, "armed": px4io.armed_intervals(u),
           "modes": px4io.mode_intervals(u)}
    _, d_att = fr.check_attitude(u, ctx)
    _, d_vib = fr.check_vibration(u, ctx)
    _, d_ekf = fr.check_estimator(u, ctx)
    _, d_act = fr.check_actuation(u, ctx)
    _, d_bat = fr.check_battery(u, ctx)
    return fr.measures(u, d_att, d_vib, d_ekf, d_act, d_bat) or None


def summarise(values: dict, out_path: str):
    print("\n" + "=" * 92)
    print(f"{'metric':16} {'n':>4} {'p10':>9} {'p50':>9} {'p75':>9} {'p90':>9} "
          f"{'p95':>9} {'p99':>9} {'max':>9}")
    print("-" * 92)
    table = {}
    for k, v in sorted(values.items()):
        v = np.array([x for x in v if np.isfinite(x)])
        if v.size < 10:
            continue
        q = np.percentile(v, [10, 50, 75, 90, 95, 99])
        table[k] = dict(n=int(v.size), p10=q[0], p50=q[1], p75=q[2], p90=q[3],
                        p95=q[4], p99=q[5], max=float(v.max()))
        print(f"{k:16} {v.size:4} {q[0]:9.3g} {q[1]:9.3g} {q[2]:9.3g} {q[3]:9.3g} "
              f"{q[4]:9.3g} {q[5]:9.3g} {v.max():9.3g}")
    json.dump(table, open(out_path, "w"), indent=1)
    json.dump({k: list(map(float, v)) for k, v in values.items()},
              open(out_path.replace(".json", "_raw.json"), "w"))
    print(f"\ndistributions written to {out_path}")

    # suggestion: WARNING at p75, FAIL at p90 (metrics where higher is worse)
    higher_worse = {"att_roll": "att_auto_warn_deg/att_auto_fail_deg",
                    "rate_p": "rate_auto_warn_dps/rate_auto_fail_dps",
                    "hf_accel": "hf_accel_warn/hf_accel_fail",
                    "vib_metric": "vib_warn/vib_fail",
                    "innov_max": "innov_warn/innov_fail",
                    "sat_pct": "sat_warn_pct/sat_fail_pct",
                    "imbalance_pct": "imbalance_warn_pct/imbalance_fail_pct",
                    "dropout_s": "dropout_warn_s/dropout_fail_s"}
    print("\n--- suggested thresholds (WARNING = fleet p75, FAIL = fleet p90)")
    print(f"{'key in THRESHOLDS':42} {'current':>14} {'suggested':>20}")
    cur = fr.THRESHOLDS
    for met, names in higher_worse.items():
        if met not in table:
            continue
        n_w, n_f = names.split("/")
        print(f"{names:42} {cur.get(n_w,'?'):>7}/{cur.get(n_f,'?'):<6} "
              f"{table[met]['p75']:>9.3g}/{table[met]['p90']:<9.3g}")
    if "vcell_min" in table:
        finite = [x for x in values["vcell_min"] if np.isfinite(x)]
        print(f"{'cell_warn_v/cell_fail_v (lower is worse)':42} "
              f"{cur['cell_warn_v']:>7}/{cur['cell_fail_v']:<6} "
              f"{np.percentile(finite, 25):>9.3g}/{table['vcell_min']['p10']:<9.3g}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", help="text file with one log id per line")
    ap.add_argument("--max", type=int, default=220)
    ap.add_argument("--out", default="out/calibration.json")
    args = ap.parse_args(argv)

    ids = [l.strip() for l in open(args.ids) if l.strip()][:args.max]
    tmp = tempfile.mkdtemp(prefix="calib_ulg_")
    values: dict[str, list] = {}
    ok = failed = 0
    try:
        for i, log_id in enumerate(ids, 1):
            p = download(log_id, tmp)
            if p is None:
                failed += 1
                print(f"[{i:3}/{len(ids)}] {log_id[:8]} download failed", flush=True)
                continue
            try:
                m = measure(p)
            except Exception as e:
                m = None
                print(f"[{i:3}/{len(ids)}] {log_id[:8]} error: {str(e)[:60]}", flush=True)
            finally:
                os.remove(p)
            if not m:
                failed += 1
                continue
            ok += 1
            for k, v in m.items():
                values.setdefault(k, []).append(v)
            print(f"[{i:3}/{len(ids)}] {log_id[:8]} ok "
                  f"({len(m)} metrics, {ok} so far)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nlogs measured: {ok} | failures: {failed}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summarise(values, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

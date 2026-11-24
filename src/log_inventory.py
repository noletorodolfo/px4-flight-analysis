"""Inventory of a PX4 .ulg log: topics, rates and data quality.

Usage: python src/log_inventory.py data/ulg/<id>/<id>.ulg [--min-hz 10]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
from pyulog import ULog

# Topics that matter for attitude tracking, vibration, EKF health and SINDy
TOPICS_OF_INTEREST = [
    "vehicle_attitude",
    "vehicle_attitude_setpoint",
    "vehicle_angular_velocity",
    "vehicle_rates_setpoint",
    "vehicle_local_position",
    "vehicle_local_position_setpoint",
    "sensor_combined",
    "sensor_accel",
    "sensor_gyro",
    "vehicle_imu_status",
    "actuator_motors",
    "actuator_outputs",
    "manual_control_setpoint",
    "estimator_status",
    "estimator_status_flags",
    "estimator_innovations",
    "vehicle_status",
    "battery_status",
]


@dataclass
class TopicStat:
    name: str
    multi_id: int
    n: int
    dt_mean: float
    hz_mean: float
    hz_median: float
    jitter_pct: float
    gap_max: float
    duration: float


def topic_stats(ulog: ULog) -> list[TopicStat]:
    stats: list[TopicStat] = []
    for d in ulog.data_list:
        t = d.data["timestamp"].astype(np.float64) * 1e-6  # s
        n = len(t)
        if n < 2:
            stats.append(TopicStat(d.name, d.multi_id, n, np.nan, np.nan, np.nan,
                                   np.nan, np.nan, 0.0))
            continue
        dt = np.diff(t)
        dt = dt[dt > 0]
        dt_mean = float(np.mean(dt)) if dt.size else np.nan
        dt_med = float(np.median(dt)) if dt.size else np.nan
        stats.append(TopicStat(
            name=d.name, multi_id=d.multi_id, n=n,
            dt_mean=dt_mean,
            hz_mean=1.0 / dt_mean if dt_mean else np.nan,
            hz_median=1.0 / dt_med if dt_med else np.nan,
            jitter_pct=float(np.std(dt) / dt_mean * 100) if dt_mean else np.nan,
            gap_max=float(np.max(dt)) if dt.size else np.nan,
            duration=float(t[-1] - t[0]),
        ))
    stats.sort(key=lambda s: (-s.n, s.name))
    return stats


def dropouts(ulog: ULog) -> tuple[int, float]:
    total_ms = sum(d.duration for d in ulog.dropouts)
    return len(ulog.dropouts), total_ms / 1000.0


def header(ulog: ULog, path: str) -> None:
    start = ulog.start_timestamp * 1e-6
    end = ulog.last_timestamp * 1e-6
    print("=" * 78)
    print(f"LOG: {path}")
    print(f"duration: {end - start:.1f} s   ({(end-start)/60:.1f} min)")
    keys = ["sys_name", "ver_sw_release", "ver_hw", "sys_autostart", "sys_mc_est_group",
            "ver_sw", "time_ref_utc"]
    info = {k: v for k, v in ulog.msg_info_dict.items() if k in keys or k.startswith("ver_")}
    for k, v in sorted(info.items()):
        print(f"  {k}: {v}")
    nd, sd = dropouts(ulog)
    print(f"  dropouts: {nd} ({sd:.3f} s lost)")
    print(f"  topics recorded: {len(ulog.data_list)}")


def print_table(stats: list[TopicStat], only: set[str] | None, min_hz: float) -> None:
    rows = [s for s in stats if (only is None or s.name in only)]
    print("-" * 78)
    print(f"{'topic':38} {'id':>2} {'n':>8} {'Hz':>7} {'jit%':>6} {'gap_s':>7}")
    print("-" * 78)
    for s in rows:
        flag = ""
        if not np.isnan(s.hz_median) and s.hz_median < min_hz:
            flag = "  <- low rate"
        if not np.isnan(s.gap_max) and s.gap_max > 1.0:
            flag += "  <- gap >1s"
        print(f"{s.name[:38]:38} {s.multi_id:>2} {s.n:>8} {s.hz_median:>7.1f} "
              f"{s.jitter_pct:>6.1f} {s.gap_max:>7.3f}{flag}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--min-hz", type=float, default=10.0)
    ap.add_argument("--all", action="store_true", help="list every topic")
    args = ap.parse_args(argv)

    ulog = ULog(args.ulg)
    header(ulog, args.ulg)
    stats = topic_stats(ulog)

    present = {s.name for s in stats}
    print("\n### Topics of interest")
    print_table(stats, set(TOPICS_OF_INTEREST), args.min_hz)
    missing = [t for t in TOPICS_OF_INTEREST if t not in present]
    if missing:
        print("\nmissing:", ", ".join(missing))

    if args.all:
        print("\n### All topics")
        print_table(stats, None, args.min_hz)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

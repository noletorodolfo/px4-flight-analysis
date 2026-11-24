"""Automatic health report for a single PX4 flight, from the .ulg log.

Writes a Markdown report with a verdict per area (attitude tracking, vibration,
estimator, actuation, battery, log quality) and a figure with the matching panels.

Thresholds are explicit choices, listed in THRESHOLDS below and printed at the
bottom of every report. They were calibrated against 209 public flights, but a
single number outside a threshold is not a defect: it is a reason to look.

Usage: python src/flight_report.py data/ulg/<id>/<id>.ulg [--out out]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402

R2D = 180.0 / np.pi
OK, WARN, FAIL, NA = "OK", "WARNING", "FAIL", "NO DATA"
COLOR = {OK: "#3C7D4E", WARN: "#C98A14", FAIL: "#B3261E", NA: "#9A9A9A"}
C_A, C_B = "#1F6FB2", "#D1670B"
INK, MUTED, GRID, BG = "#1a1a1a", "#6b6b6b", "#e3e3e0", "#fcfcfb"

# Modes where PX4 generates the attitude setpoint smoothly. In piloted modes a
# large tracking error is expected, because the setpoint is a stick step.
AUTO_MODES = {"Position", "Hold", "Mission", "Return", "Takeoff", "Land",
              "Descend", "Orbit", "Offboard", "PrecLand", "FollowMe"}

# Calibrated on 2026-11-23 against 209 public PX4 Flight Review logs (one per
# vehicle, 90-260 s, firmware >= 1.13): WARNING = fleet p75, FAIL = fleet p90.
# Distributions in out/calibration.json.
# Two deliberate exceptions, where physical meaning beats the percentile:
#   - innov_*: the fleet sits at p75 = 6.9 and p90 = 47.9, but an innovation
#     ratio above 1 means the measurement was REJECTED by the EKF. A poor fleet
#     does not make rejection acceptable, so these stay at 0.5 / 1.0.
#   - dropout_*: fleet p75 is 0.024 s, far too tight to be a useful warning;
#     adopted 0.5 / 2.0 s (fleet p90 = 2.02 s).
THRESHOLDS = {
    "att_auto_warn_deg": 2.5, "att_auto_fail_deg": 6.0,      # fleet: 2.35 / 6.02
    "rate_auto_warn_dps": 13.0, "rate_auto_fail_dps": 25.0,  # fleet: 13.5 / 25.5
    "vib_warn": 20.0, "vib_fail": 33.0,                      # fleet: 20.5 / 33.1
    "hf_accel_warn": 5.0, "hf_accel_fail": 8.0,              # fleet: 4.84 / 7.93
    "innov_warn": 0.5, "innov_fail": 1.0,                    # exception, see above
    "sat_warn_pct": 7.0, "sat_fail_pct": 35.0,               # fleet: 7.36 / 34.7
    "imbalance_warn_pct": 30.0, "imbalance_fail_pct": 54.0,  # fleet: 30.5 / 53.8
    "cell_warn_v": 3.5, "cell_fail_v": 3.3,                  # fleet: p25 3.5 / p10 3.22
    "dropout_warn_s": 0.5, "dropout_fail_s": 2.0,            # exception, see above
}


@dataclass
class Check:
    area: str
    status: str
    summary: str
    lines: list[str] = field(default_factory=list)


def worst(*st: str) -> str:
    for s in (FAIL, WARN, OK, NA):
        if s in st:
            return s
    return NA


def _armed_mask(t: np.ndarray, armed) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for a, b in armed:
        m |= (t >= a) & (t <= b)
    return m if m.any() else np.ones(len(t), dtype=bool)


# ------------------------------------------------------------------- checks
def check_attitude(u, ctx) -> tuple[Check, dict]:
    try:
        att = px4io.euler_df(u, "vehicle_attitude", "act")
        sp = px4io.euler_df(u, "vehicle_attitude_setpoint", "sp", qname="q_d")
        rate = px4io.to_df(u, "vehicle_angular_velocity")[["xyz[0]", "xyz[1]", "xyz[2]"]]
        rate.columns = ["act_p", "act_q", "act_r"]
        rsp = px4io.to_df(u, "vehicle_rates_setpoint")[["roll", "pitch", "yaw"]]
        rsp.columns = ["sp_p", "sp_q", "sp_r"]
    except Exception as e:
        return Check("Attitude tracking", NA, f"topics missing ({e})"), {}
    df = px4io.resample_join([att, sp, rate, rsp], 50.0)
    t = df.index.to_numpy()
    df = df[_armed_mask(t, ctx["armed"])]
    t = df.index.to_numpy()

    groups = {"automatic": np.zeros(len(t), bool), "piloted": np.zeros(len(t), bool)}
    for a, b, nm in ctx["modes"]:
        sel = (t >= a) & (t <= b)
        groups["automatic" if nm in AUTO_MODES else "piloted"] |= sel

    out, lines, st = {}, [], NA
    for g, m in groups.items():
        if m.sum() < 50:
            continue
        e = {}
        for name, sp_c, act_c, ang in [("roll", "sp_roll", "act_roll", True),
                                       ("pitch", "sp_pitch", "act_pitch", True),
                                       ("yaw", "sp_yaw", "act_yaw", True),
                                       ("p", "sp_p", "act_p", False),
                                       ("q", "sp_q", "act_q", False),
                                       ("r", "sp_r", "act_r", False)]:
            d = df[sp_c].to_numpy()[m] - df[act_c].to_numpy()[m]
            if ang:
                d = px4io.wrap_pi(d)
            e[name] = float(np.sqrt(np.mean(d ** 2))) * R2D
        out[g] = {"n_s": float(m.sum()) / 50.0, **e}
        lines.append(f"- **{g}** ({m.sum()/50:.0f} s): RMS roll {e['roll']:.1f}°, "
                     f"pitch {e['pitch']:.1f}°, yaw {e['yaw']:.1f}° | "
                     f"p {e['p']:.1f}, q {e['q']:.1f}, r {e['r']:.1f} °/s")
        if g == "automatic":
            att_max = max(e["roll"], e["pitch"])
            rate_max = max(e["p"], e["q"])
            s = OK
            if (att_max > THRESHOLDS["att_auto_fail_deg"]
                    or rate_max > THRESHOLDS["rate_auto_fail_dps"]):
                s = FAIL
            elif (att_max > THRESHOLDS["att_auto_warn_deg"]
                    or rate_max > THRESHOLDS["rate_auto_warn_dps"]):
                s = WARN
            st = worst(st, s)
    if "automatic" not in out:
        lines.append("- not enough time in automatic modes: RMS in a piloted mode is "
                     "not a quality criterion (the setpoint is a stick step)")
        st = NA
    summary = (f"auto RMS roll/pitch = {out['automatic']['roll']:.1f}/"
               f"{out['automatic']['pitch']:.1f}°" if "automatic" in out
               else "piloted modes only")
    return (Check("Attitude tracking", st, summary, lines),
            {"df": df, "groups": groups, "rms": out})


def check_vibration(u, ctx) -> tuple[Check, dict]:
    lines, st, data = [], NA, {}
    try:
        sc = px4io.to_df(u, "sensor_combined")
        t = sc.index.to_numpy()
        m = _armed_mask(t, ctx["armed"])
        fs = 1.0 / np.median(np.diff(t[m]))
        b, a = butter(4, min(20.0 / (fs / 2), 0.95), btype="high")
        hf = [float(np.std(filtfilt(b, a, sc[f"accelerometer_m_s2[{i}]"].to_numpy()[m])))
              for i in range(3)]
        data["hf"] = hf
        peak = max(hf)
        s = (FAIL if peak > THRESHOLDS["hf_accel_fail"] else
             WARN if peak > THRESHOLDS["hf_accel_warn"] else OK)
        st = worst(st, s)
        lines.append(f"- accelerometer above 20 Hz (RMS): "
                     f"x {hf[0]:.2f}, y {hf[1]:.2f}, z {hf[2]:.2f} m/s² "
                     f"(thresholds {THRESHOLDS['hf_accel_warn']:.0f}/"
                     f"{THRESHOLDS['hf_accel_fail']:.0f})")
    except Exception as e:
        lines.append(f"- sensor_combined unavailable ({e})")

    try:
        for mid in range(4):
            try:
                st_df = px4io.to_df(u, "vehicle_imu_status", mid)
            except Exception:
                break
            tm = st_df.index.to_numpy()
            mm = _armed_mask(tm, ctx["armed"])
            vib = st_df["accel_vibration_metric"].to_numpy()[mm]
            clip = 0
            for i in range(3):
                c = st_df.get(f"accel_clipping[{i}]")
                if c is not None:
                    v = c.to_numpy()[mm]
                    clip += int(v[-1] - v[0]) if len(v) else 0
            vmax = float(np.nanmax(vib)) if len(vib) else np.nan
            vmed = float(np.nanmedian(vib)) if len(vib) else np.nan
            # Some firmware never populates this field. A flat zero is missing
            # data, not a clean vehicle, so it must not be read as "no vibration".
            metric_ok = np.isfinite(vmax) and vmax > 0.2
            data.setdefault("vib", []).append((mid, vmed, vmax, clip, metric_ok))
            s = (FAIL if (metric_ok and vmax > THRESHOLDS["vib_fail"]) or clip > 1000 else
                 WARN if (metric_ok and vmax > THRESHOLDS["vib_warn"]) or clip > 0 else OK)
            st = worst(st, s)
            if metric_ok:
                lines.append(f"- IMU {mid}: vibration metric median {vmed:.1f}, "
                             f"peak {vmax:.1f} m/s² (thresholds "
                             f"{THRESHOLDS['vib_warn']:.0f}/"
                             f"{THRESHOLDS['vib_fail']:.0f}); clipping: {clip}")
            else:
                lines.append(f"- IMU {mid}: vibration metric not populated by this "
                             f"firmware; accelerometer clipping: {clip}")
    except Exception as e:
        lines.append(f"- vehicle_imu_status unavailable ({e})")
    if data.get("vib"):
        clip_total = sum(v[3] for v in data["vib"])
        with_metric = [v for v in data["vib"] if v[4]]
        summary = (f"peak {max(v[2] for v in with_metric):.0f} m/s²" if with_metric
                   else f"HF {max(data.get('hf', [0])):.1f} m/s²")
        if clip_total:
            summary += f", {clip_total} clipping"
    else:
        summary = "partial"
    return Check("Vibration", st, summary, lines), data


def check_estimator(u, ctx) -> tuple[Check, dict]:
    lines, st, data = [], NA, {}
    ratios = {}
    for topic in ("estimator_innovation_test_ratios", "estimator_status"):
        try:
            d = px4io.to_df(u, topic)
        except Exception:
            continue
        t = d.index.to_numpy()
        m = _armed_mask(t, ctx["armed"])
        for c in d.columns:
            if not c.endswith("test_ratio") and "test_ratios" not in topic:
                continue
            if c in ("timestamp", "timestamp_sample"):
                continue
            v = d[c].to_numpy()[m]
            v = v[np.isfinite(v)]
            if v.size == 0 or np.allclose(v, 0):
                continue
            # Ratios above 1e3 are numeric garbage from an uninitialised filter,
            # not "a very large innovation": they stay out of the statistics.
            diverged = float(np.max(v)) > 1e3
            v = v[v <= 1e3]
            if v.size == 0:
                ratios[f"{c}"] = (np.nan, np.inf, 100.0)
                continue
            ratios[f"{c}"] = (float(np.nanmedian(v)), float(np.nanmax(v)),
                              100.0 * float(np.mean(v > 1.0)))
            if diverged:
                lines.append(f"- {c}: values above 1e3 discarded "
                             f"(filter diverged or not initialised)")
        if ratios:
            data["series"] = (t[m], d, [c for c in d.columns if c in ratios])
            break
    if ratios:
        st = OK
        for k, (med, mx, frac) in sorted(ratios.items(), key=lambda kv: -kv[1][1])[:8]:
            # Gravity fusion is rejected under acceleration by design: it is
            # reported for information and never fails the flight.
            info = k.startswith("gravity")
            s = OK if info else (FAIL if frac > 1.0 else
                                 WARN if mx > THRESHOLDS["innov_warn"] else OK)
            st = worst(st, s)
            lines.append(f"- {k}: median {med:.2f}, peak {mx:.2f}, "
                         f"{frac:.2f}% of the time above 1 (rejection)"
                         + ("  *(informational: gravity is rejected under "
                            "acceleration)*" if info else ""))
        data["ratios"] = ratios
        data["critical"] = {k: v for k, v in ratios.items()
                            if not k.startswith("gravity")}
    else:
        lines.append("- no innovation ratios in this log")

    try:
        es = px4io.to_df(u, "estimator_status")
        t = es.index.to_numpy(); m = _armed_mask(t, ctx["armed"])
        ff = es["filter_fault_flags"].to_numpy()[m]
        nf = int(np.count_nonzero(ff))
        if nf:
            st = worst(st, FAIL)
            lines.append(f"- **filter_fault_flags non-zero in {nf} samples**")
        for c in [c for c in es.columns if c.startswith("reset_count")]:
            v = es[c].to_numpy()[m]
            n = int(v[-1] - v[0]) if len(v) else 0
            if n > 0:
                st = worst(st, WARN if n < 5 else FAIL)
                lines.append(f"- {c}: {n} resets during the flight")
    except Exception:
        pass
    crit = data.get("critical") or {}
    summary = (f"worst ratio (excl. gravity) {max(v[1] for v in crit.values()):.2f}"
               if crit else ("gravity only" if ratios else "no data"))
    return Check("Estimator (EKF)", st, summary, lines), data


def check_actuation(u, ctx) -> tuple[Check, dict]:
    lines, st, data = [], NA, {}
    try:
        am = px4io.to_df(u, "actuator_motors")
        cols = [c for c in am.columns if c.startswith("control[")]
    except Exception:
        try:
            am = px4io.to_df(u, "actuator_outputs")
            cols = [c for c in am.columns if c.startswith("output[")]
        except Exception as e:
            return Check("Actuation", NA, f"no motor commands ({e})"), {}
    t = am.index.to_numpy()
    m = _armed_mask(t, ctx["armed"])
    U = np.column_stack([am[c].to_numpy() for c in cols])[m]
    with np.errstate(all="ignore"):
        filled = ~np.all(np.isnan(U), axis=0)
        spread = np.zeros(U.shape[1])
        if filled.any():
            spread[filled] = np.nanstd(U[:, filled], axis=0)
    keep = filled & (spread > 1e-6)
    U = U[:, keep]
    if U.size == 0:
        return Check("Actuation", NA, "constant/NaN commands"), {}
    if np.nanmax(U) > 10:                      # PWM in us -> normalise
        U = (U - 1000.0) / 1000.0
    sat = 100.0 * float(np.mean(np.any((U <= 0.02) | (U >= 0.95), axis=1)))
    mean_cmd = np.nanmean(U, axis=0)
    # With a near-zero mean command (vehicle on the ground) the ratio explodes.
    imbalance = (100.0 * float((mean_cmd.max() - mean_cmd.min()) / mean_cmd.mean())
                 if float(mean_cmd.mean()) > 0.05 else np.nan)
    if np.isfinite(imbalance) and imbalance > 300.0:
        imbalance = np.nan     # above this the measurement is broken, not the vehicle
    data["U"] = U; data["mean_cmd"] = mean_cmd; data["t"] = t[m]
    s_sat = (FAIL if sat > THRESHOLDS["sat_fail_pct"] else
             WARN if sat > THRESHOLDS["sat_warn_pct"] else OK)
    s_imb = (NA if not np.isfinite(imbalance) else
             FAIL if imbalance > THRESHOLDS["imbalance_fail_pct"] else
             WARN if imbalance > THRESHOLDS["imbalance_warn_pct"] else OK)
    st = worst(s_sat, s_imb)
    lines.append(f"- allocator saturation: {sat:.1f}% of armed time with some motor "
                 f"<= 0.02 or >= 0.95 (thresholds {THRESHOLDS['sat_warn_pct']:.0f}/"
                 f"{THRESHOLDS['sat_fail_pct']:.0f}%)")
    per_motor = ", ".join(f"{v:.3f}" for v in mean_cmd)
    if np.isfinite(imbalance):
        lines.append(f"- mean command per motor: {per_motor} -> imbalance "
                     f"{imbalance:.1f}% (thresholds "
                     f"{THRESHOLDS['imbalance_warn_pct']:.0f}/"
                     f"{THRESHOLDS['imbalance_fail_pct']:.0f}%)")
    else:
        lines.append(f"- mean command per motor: {per_motor} -> imbalance not "
                     f"evaluated (mean thrust below 0.05, or ratio out of range)")
    if np.isfinite(imbalance) and imbalance > THRESHOLDS["imbalance_warn_pct"]:
        lines.append("  - high imbalance usually means a shifted CG, a bent arm, "
                     "a damaged propeller or a weak motor")
    lines.append(f"- mean thrust margin: {100*(1-np.nanmean(U)):.0f}% "
                 f"(mean command {np.nanmean(U):.3f})")
    imb_text = f"{imbalance:.0f}%" if np.isfinite(imbalance) else "n/a"
    return Check("Actuation", st, f"saturation {sat:.0f}%, imbalance {imb_text}",
                 lines), data


def check_battery(u, ctx) -> tuple[Check, dict]:
    try:
        bs = px4io.to_df(u, "battery_status")
    except Exception as e:
        return Check("Battery", NA, f"no battery_status ({e})"), {}
    t = bs.index.to_numpy(); m = _armed_mask(t, ctx["armed"])
    vcol = "voltage_filtered_v" if "voltage_filtered_v" in bs.columns else "voltage_v"
    if vcol not in bs.columns:
        return Check("Battery", NA, "no voltage in this log"), {}
    v = bs[vcol].to_numpy()[m]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Check("Battery", NA, "invalid voltage"), {}
    cells = bs["cell_count"].to_numpy()[m] if "cell_count" in bs.columns else np.array([0])
    nc = int(np.nanmax(cells)) if np.isfinite(np.nanmax(cells)) and np.nanmax(cells) > 0 else 0

    # A wrong or zeroed cell_count is common, so validate against the physical
    # Li-ion window and infer the count when the field cannot be trusted.
    def plausible(n):
        return n > 0 and 2.5 <= float(np.nanmedian(v)) / n <= 4.35

    if not plausible(nc):
        cand = [n for n in range(1, 15) if plausible(n)]
        nc = cand[0] if cand else 0
    cur = bs["current_a"].to_numpy()[m] if "current_a" in bs.columns else np.array([np.nan])
    data = {"t": t[m], "v": v, "i": cur, "nc": nc}
    lines, st = [], OK
    if nc:
        vc = v / nc
        vc = vc[(vc > 1.5) & (vc < 4.6)]
        if vc.size == 0:
            return Check("Battery", NA, "voltage outside physical range"), {}
        lines.append(f"- {nc}S: start {vc[0]:.2f} V/cell, end {vc[-1]:.2f}, "
                     f"minimum {np.nanmin(vc):.2f} "
                     f"(thresholds {THRESHOLDS['cell_warn_v']}/{THRESHOLDS['cell_fail_v']})")
        st = (FAIL if np.nanmin(vc) < THRESHOLDS["cell_fail_v"] else
              WARN if np.nanmin(vc) < THRESHOLDS["cell_warn_v"] else OK)
        sag = float(np.nanmax(vc) - np.nanmin(vc))
        lines.append(f"- voltage sag under load: {sag:.2f} V/cell")
    else:
        lines.append(f"- voltage: start {v[0]:.2f} V, end {v[-1]:.2f}, "
                     f"minimum {np.nanmin(v):.2f} (cell count unavailable)")
        st = NA
    if np.isfinite(cur).any():
        lines.append(f"- current: mean {np.nanmean(cur):.1f} A, peak {np.nanmax(cur):.1f} A")
    if "discharged_mah" in bs.columns:
        dm = bs["discharged_mah"].to_numpy()[m]
        lines.append(f"- consumed: {np.nanmax(dm):.0f} mAh")
    if "remaining" in bs.columns:
        rem = bs["remaining"].to_numpy()[m]
        lines.append(f"- estimated remaining at landing: {100*np.nanmin(rem):.0f}%")
    return Check("Battery", st,
                 f"min {np.nanmin(vc):.2f} V/cell" if nc else "no cell count",
                 lines), data


def check_log(u, ctx) -> tuple[Check, dict]:
    nd = len(u.dropouts)
    sd = sum(d.duration for d in u.dropouts) / 1000.0
    st = (FAIL if sd > THRESHOLDS["dropout_fail_s"] else
          WARN if sd > THRESHOLDS["dropout_warn_s"] else OK)
    lines = [f"- dropouts: {nd} ({sd:.2f} s lost; thresholds "
             f"{THRESHOLDS['dropout_warn_s']:.1f}/{THRESHOLDS['dropout_fail_s']:.1f} s)",
             f"- topics recorded: {len(u.data_list)}"]
    msgs = [m for m in u.logged_messages if m.log_level <= 4]
    if msgs:
        st = worst(st, WARN)
        lines.append(f"- {len(msgs)} warning/error messages from the firmware:")
        for m in msgs[:8]:
            lines.append(f"  - t={m.timestamp*1e-6 - u.start_timestamp*1e-6:.1f}s: "
                         f"{m.message[:90]}")
    return Check("Log quality", st, f"{nd} dropouts, {sd:.1f} s", lines), {}


# ------------------------------------------------- numbers for calibration
def measures(u, d_att, d_vib, d_ekf, d_act, d_bat) -> dict:
    """Pull, from the same checks, the numbers that become thresholds.

    Used by the report (to compare against the fleet) and by
    src/calibrate_thresholds.py (to build the fleet distribution).
    """
    m: dict = {}
    a = (d_att or {}).get("rms", {}).get("automatic")
    if a:
        m.update(att_roll=a["roll"], att_pitch=a["pitch"], att_yaw=a["yaw"],
                 rate_p=a["p"], rate_q=a["q"], rate_r=a["r"], t_auto=a["n_s"])
    if (d_vib or {}).get("hf"):
        m["hf_accel"] = float(max(d_vib["hf"]))
    if (d_vib or {}).get("vib"):
        with_metric = [v for v in d_vib["vib"] if v[4]]
        if with_metric:
            m["vib_metric"] = float(max(v[2] for v in with_metric))
        m["clipping"] = float(sum(v[3] for v in d_vib["vib"]))
    crit = (d_ekf or {}).get("critical") or {}
    finite = [v for v in crit.values() if np.isfinite(v[1])]
    if finite:
        m["innov_max"] = float(max(v[1] for v in finite))
        m["innov_rej_pct"] = float(max(v[2] for v in finite))
    if (d_act or {}).get("mean_cmd") is not None and len((d_act or {}).get("mean_cmd", [])):
        U, mc = d_act["U"], d_act["mean_cmd"]
        m["sat_pct"] = 100.0 * float(np.mean(np.any((U <= 0.02) | (U >= 0.95), axis=1)))
        if float(np.nanmean(mc)) > 0.05:
            imb = 100.0 * float((mc.max() - mc.min()) / mc.mean())
            if imb <= 300.0:
                m["imbalance_pct"] = imb
        m["mean_motor"] = float(np.nanmean(U))
    if (d_bat or {}).get("nc"):
        vc = d_bat["v"] / d_bat["nc"]
        vc = vc[(vc > 1.5) & (vc < 4.6)]
        if vc.size:
            m["vcell_min"] = float(np.nanmin(vc))
    m["dropout_s"] = sum(d.duration for d in u.dropouts) / 1000.0
    return m


# metric -> (label, higher_is_worse)
FLEET_LABELS = {
    "att_roll": ("roll attitude error", True),
    "att_pitch": ("pitch attitude error", True),
    "rate_p": ("roll-rate error", True),
    "rate_q": ("pitch-rate error", True),
    "hf_accel": ("accelerometer above 20 Hz", True),
    "vib_metric": ("vibration metric", True),
    "clipping": ("accelerometer clipping", True),
    "innov_max": ("worst innovation ratio", True),
    "sat_pct": ("allocator saturation", True),
    "imbalance_pct": ("motor imbalance", True),
    "dropout_s": ("dropouts", True),
    "vcell_min": ("minimum cell voltage", False),
}


def fleet_percentiles(med: dict, path="out/calibration_raw.json"):
    """Place each measurement inside the measured public-fleet distribution."""
    if not os.path.exists(path):
        return None, 0
    try:
        raw = json.load(open(path))
    except Exception:
        return None, 0
    n = max((len(v) for v in raw.values()), default=0)
    out = []
    for k, (label, higher_worse) in FLEET_LABELS.items():
        if k not in med or k not in raw:
            continue
        sample = np.array([x for x in raw[k] if np.isfinite(x)])
        if sample.size < 20:
            continue
        pct = 100.0 * float(np.mean(sample <= med[k]))
        if not higher_worse:
            pct = 100.0 - pct
        out.append((label, med[k], pct, sample.size))
    return out, n


# ------------------------------------------------------------------ output
def figure_panel(path_png, name, ctx, d_att, d_vib, d_ekf, d_act, d_bat, checks):
    fig = plt.figure(figsize=(14, 11), layout="constrained")
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.35, 1, 1, 1])

    # status strip
    axs = fig.add_subplot(gs[0, :]); axs.set_facecolor(BG); axs.axis("off")
    for i, c in enumerate(checks):
        x = i / len(checks)
        axs.add_patch(plt.Rectangle((x, 0.25), 1 / len(checks) - 0.01, 0.5,
                                    color=COLOR[c.status], alpha=0.85))
        axs.text(x + 0.006, 0.66, c.area[:24], fontsize=8, color="white", va="center")
        axs.text(x + 0.006, 0.48, c.status, fontsize=7.5, color="white", va="center",
                 weight="bold")
        axs.text(x + 0.006, 0.32, c.summary[:30], fontsize=6.8, color="white",
                 va="center")
    axs.set_xlim(0, 1); axs.set_ylim(0, 1)

    arm = ctx["armed"]
    xlim = (arm[0][0] - 5, arm[-1][1] + 5) if arm else None

    def style(ax, title, ylabel):
        ax.set_facecolor(BG); ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_ylabel(ylabel, fontsize=8, color=INK)

    # attitude
    ax = fig.add_subplot(gs[1, 0]); style(ax, "attitude error", "deg")
    if d_att:
        df = d_att["df"]; t = df.index.to_numpy()
        for c, lab, col in [("roll", "roll", C_A), ("pitch", "pitch", C_B)]:
            e = px4io.wrap_pi(df[f"sp_{c}"].to_numpy() - df[f"act_{c}"].to_numpy()) * R2D
            ax.plot(t, e, lw=0.7, color=col, label=lab)
        auto = d_att["groups"]["automatic"]
        if auto.any():
            ax.fill_between(t, -180, 180, where=auto, color="#1F6FB2", alpha=0.07,
                            label="automatic mode")
        ax.set_ylim(-60, 60); ax.legend(fontsize=7, frameon=False, ncol=3)

    # vibration
    ax = fig.add_subplot(gs[1, 1]); style(ax, "vibration (accel_vibration_metric)",
                                          "m/s²")
    try:
        for v in d_vib.get("vib", []):
            if not v[4]:
                continue
            s = px4io.to_df(ctx["u"], "vehicle_imu_status", v[0])
            ax.plot(s.index, s["accel_vibration_metric"], lw=0.9, label=f"IMU {v[0]}")
        if ax.get_legend_handles_labels()[0]:
            ax.axhline(THRESHOLDS["vib_warn"], color=COLOR[WARN], lw=1, ls="--")
            ax.axhline(THRESHOLDS["vib_fail"], color=COLOR[FAIL], lw=1, ls="--")
            ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
            if xlim:
                ax.set_xlim(*xlim)
        else:
            ax.text(0.5, 0.5, "metric not populated by this firmware",
                    transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
    except Exception:
        pass

    # ekf
    ax = fig.add_subplot(gs[2, 0]); style(ax, "EKF innovation test ratios", "-")
    if d_ekf.get("series"):
        tt, dd, cs = d_ekf["series"]
        for c in cs[:6]:
            ax.plot(dd.index, dd[c], lw=0.8, label=c.replace("_test_ratio", ""))
        ax.axhline(1.0, color=COLOR[FAIL], lw=1, ls="--")
        if xlim:
            ax.set_xlim(*xlim)
            ax.relim(); ax.autoscale_view(scalex=False)
        ax.set_ylim(0, max(1.4, min(4.0, ax.get_ylim()[1])))
        ax.legend(fontsize=6.5, frameon=False, ncol=3)

    # actuation
    ax = fig.add_subplot(gs[2, 1]); style(ax, "command per motor", "command [0..1]")
    if d_act:
        U, t = d_act["U"], d_act["t"]
        for i in range(U.shape[1]):
            ax.plot(t, U[:, i], lw=0.6, label=f"m{i}")
        ax.axhline(0.95, color=COLOR[WARN], lw=1, ls="--")
        ax.legend(fontsize=6.5, frameon=False, ncol=4)

    # battery
    ax = fig.add_subplot(gs[3, 0]); style(ax, "battery", "V/cell")
    if d_bat and d_bat.get("nc"):
        ax.plot(d_bat["t"], d_bat["v"] / d_bat["nc"], lw=1.1, color=C_A)
        ax.axhline(THRESHOLDS["cell_warn_v"], color=COLOR[WARN], lw=1, ls="--")
        ax.axhline(THRESHOLDS["cell_fail_v"], color=COLOR[FAIL], lw=1, ls="--")
        ax.set_xlabel("time [s]", fontsize=8, color=INK)

    # imbalance
    ax = fig.add_subplot(gs[3, 1]); style(ax, "mean command per motor",
                                          "mean command [0..1]")
    if d_act:
        mc = d_act["mean_cmd"]
        ax.bar(np.arange(len(mc)), mc, color=C_A)
        ax.axhline(float(np.mean(mc)), color=C_B, lw=1.2, ls="--", label="mean")
        ax.set_xticks(np.arange(len(mc)))
        ax.set_xticklabels([f"m{i}" for i in range(len(mc))])
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"Flight report — {name}", fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(path_png, dpi=120)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    u = px4io.load(args.ulg)
    name = os.path.basename(args.ulg).replace(".ulg", "")
    ctx = {"u": u, "armed": px4io.armed_intervals(u), "modes": px4io.mode_intervals(u)}

    c_att, d_att = check_attitude(u, ctx)
    c_vib, d_vib = check_vibration(u, ctx)
    c_ekf, d_ekf = check_estimator(u, ctx)
    c_act, d_act = check_actuation(u, ctx)
    c_bat, d_bat = check_battery(u, ctx)
    c_log, _ = check_log(u, ctx)
    checks = [c_att, c_vib, c_ekf, c_act, c_bat, c_log]

    dur = (u.last_timestamp - u.start_timestamp) * 1e-6
    t_arm = sum(b - a for a, b in ctx["armed"])
    modes = {}
    for a, b, nm in ctx["modes"]:
        modes[nm] = modes.get(nm, 0.0) + (b - a)

    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"report_{name[:8]}.png")
    md = os.path.join(args.out, f"report_{name[:8]}.md")
    figure_panel(png, name[:8], ctx, d_att, d_vib, d_ekf, d_act, d_bat, checks)

    L = []
    L.append(f"# Flight report — `{name[:8]}`\n")
    L.append(f"- vehicle: {u.msg_info_dict.get('ver_hw','?')} "
             f"({u.msg_info_dict.get('ver_hw_subtype','')})")
    L.append(f"- firmware: {u.msg_info_dict.get('ver_sw_branch','?')} "
             f"`{str(u.msg_info_dict.get('ver_sw',''))[:10]}`")
    L.append(f"- log duration: {dur:.0f} s | armed: {t_arm:.0f} s")
    L.append("- time per mode: " + ", ".join(
        f"{k} {v:.0f} s" for k, v in sorted(modes.items(), key=lambda kv: -kv[1])
        if v >= 1))
    L.append("")
    overall = worst(*[c.status for c in checks])
    L.append(f"**Overall verdict: {overall}**\n")
    L.append("| area | status | summary |")
    L.append("|---|---|---|")
    for c in checks:
        L.append(f"| {c.area} | {c.status} | {c.summary} |")
    L.append("")
    for c in checks:
        L.append(f"## {c.area} — {c.status}\n")
        L.extend(c.lines or ["- no data"])
        L.append("")

    med = measures(u, d_att, d_vib, d_ekf, d_act, d_bat)
    fleet, n_fleet = fleet_percentiles(med)
    if fleet:
        L.append("## Position in the public fleet\n")
        L.append(f"Reference sample: public PX4 Flight Review logs, one per vehicle "
                 f"(n up to {n_fleet}). The percentile says how much of the fleet is "
                 f"in an equal or better situation than this flight.\n")
        L.append("| metric | value | percentile (100 = worst in fleet) |")
        L.append("|---|---|---|")
        for label, val, pct, nn in sorted(fleet, key=lambda r: -r[2]):
            mark = (" **worse than most**" if pct >= 90 else
                    " above median" if pct >= 50 else "")
            L.append(f"| {label} | {val:.3g} | {pct:.0f}th (n={nn}){mark} |")
        L.append("")

    L.append("---")
    L.append("Thresholds used (calibrated against 209 public flights, not a standard): "
             + ", ".join(f"`{k}`={v}" for k, v in THRESHOLDS.items()))
    L.append(f"\n![panel]({os.path.basename(png)})")
    open(md, "w").write("\n".join(L) + "\n")

    print(f"report: {md}")
    print(f"figure: {png}")
    print(f"\noverall verdict: {overall}")
    for c in checks:
        print(f"  {c.status:8} {c.area:22} {c.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

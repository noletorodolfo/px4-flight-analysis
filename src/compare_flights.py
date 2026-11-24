"""Compare two or more flights: what changed in the configuration, and what
changed in the flight.

Answers the flight-test question: "I touched a gain / swapped a propeller /
changed firmware — did it help?"

Three parts:
  1. a diff of the PX4 parameters stored in the logs (what actually changed);
  2. performance metrics measured **only in automatic modes**, where PX4
     generates the setpoint itself and the comparison means something;
  3. uncertainty: every metric carries a confidence interval from a 10 s block
     bootstrap, and a difference is only called significant when the interval of
     the difference excludes zero.

Significant does not mean caused by the parameter change: wind, payload and
piloting also change between flights. That is why the report prints
comparability indicators (command aggressiveness, mean thrust, usable time).

Usage:
  python src/compare_flights.py A.ulg B.ulg [C.ulg ...] [--name labels]
  python src/compare_flights.py A.ulg --split 300   # validation: split one flight
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import freq_id as fq  # noqa: E402
from flight_report import AUTO_MODES  # noqa: E402

R2D = 180.0 / np.pi
C = ["#1F6FB2", "#D1670B", "#5B8C3E", "#8E5AA8"]
INK, MUTED, GRID, BG = "#1a1a1a", "#6b6b6b", "#e3e3e0", "#fcfcfb"
HZ = 50.0
BLOCK_S = 10.0
NBOOT = 400

# Metrics whose value depends on how hard the aircraft was flown.
CONDITION_SENSITIVE = {"err_roll", "err_pitch", "err_p", "err_q", "sat"}
# Metrics that do not scale with aggressiveness: cleaner evidence.
ROBUST = {"err_p_norm", "err_q_norm", "lag_p", "lag_q", "K_p", "K_q", "band_p"}

TUNING_PREFIXES = ("MC_", "MPC_", "ATT_", "IMU_", "CA_", "THR_", "MOT_", "PWM_",
                   "RATE_", "EKF2_", "SENS_")

# metric -> (label, unit, lower_is_better)
METRICS = {
    "err_roll": ("roll attitude error (RMS)", "deg", True),
    "err_pitch": ("pitch attitude error (RMS)", "deg", True),
    "err_p": ("roll-rate error (RMS)", "deg/s", True),
    "err_q": ("pitch-rate error (RMS)", "deg/s", True),
    "err_p_norm": ("roll-rate error normalised by command", "-", True),
    "err_q_norm": ("pitch-rate error normalised by command", "-", True),
    "lag_p": ("tracking lag, roll", "ms", True),
    "lag_q": ("tracking lag, pitch", "ms", True),
    "vib_hf": ("vibration above 20 Hz (worst axis)", "m/s²", True),
    "sat": ("allocator saturation", "%", True),
    "mean_motor": ("mean motor command", "0..1", None),
    "exc_p": ("aggressiveness: std of roll-rate command", "deg/s", None),
    # Plant properties from the frequency domain: these change with hardware,
    # not with controller gains.
    "K_p": ("plant effectiveness, roll (frequency)", "rad/s² per tau", None),
    "K_q": ("plant effectiveness, pitch (frequency)", "rad/s² per tau", None),
    "band_p": ("bandwidth with coherence >= 0.6, roll", "Hz", None),
}

NO_CI = {"vib_hf", "K_p", "K_q", "band_p"}


def auto_window(u, t):
    m = np.zeros(len(t), dtype=bool)
    for a, b, nm in px4io.mode_intervals(u):
        if nm in AUTO_MODES:
            m |= (t >= a) & (t <= b)
    arm = np.zeros(len(t), dtype=bool)
    for a, b in px4io.armed_intervals(u):
        arm |= (t >= a) & (t <= b)
    return m & arm


def lag_ms(sp, act, hz, max_ms=400, corr_min=0.5):
    """Lag that maximises cross-correlation between command and response.

    Returns NaN when the peak correlation is weak (< corr_min): in that case the
    "lag" would only be where noise happened to line up, not a real delay.
    """
    sp = sp - sp.mean(); act = act - act.mean()
    n = int(max_ms * 1e-3 * hz)
    if sp.std() < 1e-9 or act.std() < 1e-9:
        return np.nan
    cs = []
    for k in range(0, n + 1):
        a = sp[:len(sp) - k] if k else sp
        b = act[k:] if k else act
        cs.append(float(np.corrcoef(a, b)[0, 1]))
    cs = np.array(cs)
    k = int(np.argmax(cs))
    if cs[k] < corr_min:
        return np.nan
    # parabolic interpolation at the peak: resolution below the sample step
    if 0 < k < len(cs) - 1:
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = (y0 - 2 * y1 + y2)
        k = k + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
    return float(k) / hz * 1000.0


def load_flight(path):
    u = px4io.load(path, ["vehicle_attitude", "vehicle_attitude_setpoint",
                          "vehicle_angular_velocity", "vehicle_rates_setpoint",
                          "vehicle_status", "actuator_motors", "actuator_outputs",
                          "sensor_combined", "vehicle_torque_setpoint",
                          "vehicle_thrust_setpoint"])
    att = px4io.euler_df(u, "vehicle_attitude", "act")
    sp = px4io.euler_df(u, "vehicle_attitude_setpoint", "sp", qname="q_d")
    rate = px4io.to_df(u, "vehicle_angular_velocity")[["xyz[0]", "xyz[1]", "xyz[2]"]]
    rate.columns = ["act_p", "act_q", "act_r"]
    rsp = px4io.to_df(u, "vehicle_rates_setpoint")[["roll", "pitch", "yaw"]]
    rsp.columns = ["sp_p", "sp_q", "sp_r"]
    frames = [att, sp, rate, rsp]
    try:
        am = px4io.to_df(u, "actuator_motors")
        cols = [c for c in am.columns if c.startswith("control[")]
        am = am[cols].fillna(0.0)
        am = am.loc[:, am.std() > 1e-6]
        am.columns = [f"m{i}" for i in range(am.shape[1])]
        frames.append(am)
        mot = list(am.columns)
    except Exception:
        mot = []
    df = px4io.resample_join(frames, HZ)
    t = df.index.to_numpy()
    df = df[auto_window(u, t)]

    # vibration measured on sensor_combined itself, not on the resampled grid
    vib = np.nan
    try:
        sc = px4io.to_df(u, "sensor_combined")
        ts = sc.index.to_numpy()
        ms = auto_window(u, ts)
        if ms.sum() > 500:
            fs = 1.0 / np.median(np.diff(ts[ms]))
            b, a = butter(4, min(20.0 / (fs / 2), 0.95), btype="high")
            vib = max(float(np.std(filtfilt(b, a, sc[f"accelerometer_m_s2[{i}]"]
                                            .to_numpy()[ms]))) for i in range(3))
    except Exception:
        pass

    # frequency response: separates a plant property from a loop setting
    freq = {}
    try:
        tq = px4io.to_df(u, "vehicle_torque_setpoint")[["xyz[0]", "xyz[1]"]]
        tq.columns = ["tau_x", "tau_y"]
        rt = px4io.to_df(u, "vehicle_angular_velocity")[["xyz[0]", "xyz[1]"]]
        rt.columns = ["pp", "qq"]
        d2 = px4io.resample_join([tq, rt], HZ)
        t2 = d2.index.to_numpy()
        d2 = d2[auto_window(u, t2)]
        if len(d2) > 8 * HZ * 2:
            from scipy.signal import coherence as _coh, csd as _csd, welch as _welch
            nper = int(8 * HZ)
            for axis, (cu, cy) in {"p": ("tau_x", "pp"), "q": ("tau_y", "qq")}.items():
                x = d2[cu].to_numpy(); y = d2[cy].to_numpy()
                f_, Pxx = _welch(x, fs=HZ, nperseg=nper)
                _, Pxy = _csd(x, y, fs=HZ, nperseg=nper)
                _, coh = _coh(x, y, fs=HZ, nperseg=nper)
                H = Pxy / np.where(Pxx == 0, np.nan, Pxx)
                b = (f_ > 0.05) & (f_ < 10) & (coh >= 0.6)
                if b.sum() >= 5:
                    best = None
                    for td in np.arange(0.0, 0.201, 0.01):
                        a_, K_, r_, _ = fq.fit_first_order(
                            f_[b], H[b] * np.exp(1j * 2 * np.pi * f_[b] * td), coh[b])
                        if best is None or r_ > best[2]:
                            best = (a_, K_, r_, td)
                    freq[f"K_{axis}"] = abs(best[1])
                    if axis == "p":
                        freq["band_p"] = float(f_[b].max() - f_[b].min())
    except Exception:
        pass
    return u, df, mot, vib, freq


def _contiguous_runs(idx, min_len):
    """Split an index array into runs of consecutive samples."""
    if len(idx) == 0:
        return []
    cuts = np.where(np.diff(idx) != 1)[0] + 1
    return [r for r in np.split(np.asarray(idx), cuts) if len(r) >= min_len]


def _lag_over_runs(df, idx, axis, hz, seg_s=BLOCK_S):
    """Median tracking lag over contiguous segments.

    Cross-correlation only means something on a continuous stretch of time. The
    block bootstrap concatenates shuffled blocks, so computing the lag over the
    concatenation compares a different statistic to the point estimate — which
    is how a point estimate ended up sitting on the edge of its own interval.
    Here the lag is estimated inside each contiguous segment and the median is
    taken, so the point estimate and the bootstrap measure the same thing.
    """
    seg = int(seg_s * hz)
    vals = []
    for run in _contiguous_runs(idx, seg):
        for s0 in range(0, len(run) - seg + 1, seg):
            sub = run[s0:s0 + seg]
            d = df.iloc[sub]
            v = lag_ms(d[f"sp_{axis}"].to_numpy(), d[f"act_{axis}"].to_numpy(), hz)
            if np.isfinite(v):
                vals.append(v)
    return float(np.median(vals)) if vals else np.nan


def block_metrics(df, mot, vib, idx, freq=None):
    """Metrics over a subset of samples (one block or a set of blocks)."""
    d = df.iloc[idx]
    out = {}
    for name, a, b in [("err_roll", "sp_roll", "act_roll"),
                       ("err_pitch", "sp_pitch", "act_pitch")]:
        e = px4io.wrap_pi(d[a].to_numpy() - d[b].to_numpy()) * R2D
        out[name] = float(np.sqrt(np.mean(e ** 2)))
    for axis in ("p", "q"):
        e = (d[f"sp_{axis}"].to_numpy() - d[f"act_{axis}"].to_numpy()) * R2D
        s = float(np.std(d[f"sp_{axis}"].to_numpy()) * R2D)
        out[f"err_{axis}"] = float(np.sqrt(np.mean(e ** 2)))
        # dividing by a near-zero command inflates the metric: require 3 deg/s
        out[f"err_{axis}_norm"] = out[f"err_{axis}"] / s if s >= 3.0 else np.nan
        out[f"lag_{axis}"] = _lag_over_runs(df, idx, axis, HZ)
    out["exc_p"] = float(np.std(d["sp_p"].to_numpy()) * R2D)
    out["vib_hf"] = vib
    for k in ("K_p", "K_q", "band_p"):
        out[k] = (freq or {}).get(k, np.nan)
    if mot:
        U = d[[f"m{i}" for i in range(len(mot))]].to_numpy()
        out["sat"] = 100.0 * float(np.mean(np.any((U <= 0.02) | (U >= 0.95), axis=1)))
        out["mean_motor"] = float(np.nanmean(U))
    else:
        out["sat"] = np.nan
        out["mean_motor"] = np.nan
    return out


def bootstrap(df, mot, vib, freq=None, nboot=NBOOT, rng=None):
    """Block bootstrap: returns (point estimate, bootstrap samples) per metric."""
    rng = rng or np.random.default_rng(0)
    n = len(df)
    nb = max(2, int(n / (BLOCK_S * HZ)))
    blocks = np.array_split(np.arange(n), nb)
    base = block_metrics(df, mot, vib, np.arange(n), freq)
    samples = {k: [] for k in base}
    for _ in range(nboot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        m = block_metrics(df, mot, vib, idx, freq)
        for k, v in m.items():
            samples[k].append(v)
    return base, {k: np.array(v, dtype=float) for k, v in samples.items()}


def parameter_diff(ulogs):
    ps = [u.initial_parameters for u in ulogs]
    keys = set().union(*[set(p) for p in ps])
    noise = ("CAL_", "COM_FLIGHT_UUID", "LND_FLIGHT_T", "RC1_", "RC2_", "RC3_",
             "RC4_", "RC5_", "RC6_", "RC7_", "RC8_", "RC9_", "RC1", "SENS_BOARD")
    tuning, other = [], []
    for k in sorted(keys):
        vals = [p.get(k) for p in ps]
        if len({str(v) for v in vals}) == 1:
            continue
        (tuning if k.startswith(TUNING_PREFIXES) else other).append((k, vals))
    other = [(k, v) for k, v in other if not k.startswith(noise)]
    return tuning, other


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulgs", nargs="+")
    ap.add_argument("--name", nargs="*", default=None)
    ap.add_argument("--split", type=float, default=None,
                    help="split ONE log into two pseudo-flights at this time "
                         "(used to validate the statistics)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    ulogs, dfs, mots, vibs, freqs, names = [], [], [], [], [], []
    if args.split is not None:
        u, df, mot, vib, freq = load_flight(args.ulgs[0])
        base = os.path.basename(args.ulgs[0])[:8]
        for lab, sel in [(f"{base}[:{args.split:.0f}s]", df.index < args.split),
                         (f"{base}[{args.split:.0f}s:]", df.index >= args.split)]:
            ulogs.append(u); dfs.append(df[sel]); mots.append(mot); vibs.append(vib)
            freqs.append(freq); names.append(lab)
    else:
        for i, p in enumerate(args.ulgs):
            u, df, mot, vib, freq = load_flight(p)
            ulogs.append(u); dfs.append(df); mots.append(mot); vibs.append(vib)
            freqs.append(freq)
            names.append(args.name[i] if args.name and i < len(args.name)
                         else os.path.basename(p)[:8])

    for name, df in zip(names, dfs):
        if len(df) < BLOCK_S * HZ * 2:
            print(f"ERROR: {name} has {len(df)/HZ:.0f} s in automatic modes, "
                  f"too little to compare")
            return 1

    rng = np.random.default_rng(42)
    bases, boots = [], []
    for df, mot, vib, fq_ in zip(dfs, mots, vibs, freqs):
        b, s = bootstrap(df, mot, vib, fq_, rng=rng)
        bases.append(b); boots.append(s)

    uuids = {u.msg_info_dict.get("sys_uuid", "?") for u in ulogs}
    same_vehicle = len(uuids) == 1 or args.split is not None

    L = ["# Flight comparison\n"]
    L.append("| flight | vehicle | firmware | time in automatic modes |")
    L.append("|---|---|---|---|")
    for name, u, df in zip(names, ulogs, dfs):
        L.append(f"| {name} | {u.msg_info_dict.get('ver_hw','?')} "
                 f"`{str(u.msg_info_dict.get('sys_uuid',''))[-8:]}` | "
                 f"{u.msg_info_dict.get('ver_sw_branch','?')} | {len(df)/HZ:.0f} s |")
    L.append("")
    if not same_vehicle:
        L.append("> **Warning: these logs are not from the same vehicle** "
                 "(`sys_uuid` differs). The comparison below mixes configuration "
                 "differences with airframe differences.\n")

    # --------------------------------------------------- parameter diff
    if args.split is None:
        tuning, other = parameter_diff(ulogs)
        L.append("## What changed in the configuration\n")
        if tuning:
            L.append("Control / estimation / actuation parameters:\n")
            L.append("| parameter | " + " | ".join(names) + " |")
            L.append("|---" * (len(names) + 1) + "|")
            for k, vals in tuning:
                L.append(f"| `{k}` | " + " | ".join(
                    f"{v:.4g}" if isinstance(v, float) else str(v) for v in vals) + " |")
        else:
            L.append("- no control or estimation parameter changed between flights")
        L.append(f"\n- other parameters that differ (telemetry, RC, etc.): {len(other)}")
        if other[:10]:
            L.append("  - " + ", ".join(f"`{k}`" for k, _ in other[:10]))
        L.append("")

    # ------------------------------------------------------- comparison
    L.append("## What changed in the flight\n")
    L.append("Measured in automatic modes only. CI = 90 % interval from a 10 s "
             "block bootstrap.\n")
    header = "| metric | " + " | ".join(names) + " |"
    if len(names) == 2:
        header = "| metric | " + " | ".join(names) + " | difference (90% CI) | |"
    L.append(header)
    L.append("|---" * (len(header.split("|")) - 2) + "|")

    findings = []
    for k, (label, unit, lower_better) in METRICS.items():
        if all(not np.isfinite(b.get(k, np.nan)) for b in bases):
            continue
        no_ci = k in NO_CI
        cells = []
        for b, s in zip(bases, boots):
            v = b.get(k, np.nan)
            if no_ci:
                cells.append(f"{v:.3g}")
                continue
            lo, hi = (np.nanpercentile(s[k], 5), np.nanpercentile(s[k], 95)) \
                if k in s and np.isfinite(s[k]).any() else (np.nan, np.nan)
            cells.append(f"{v:.3g} [{lo:.3g}–{hi:.3g}]")
        row = f"| {label} [{unit}] | " + " | ".join(cells) + " |"
        if len(names) == 2 and not no_ci and k in boots[0] and k in boots[1]:
            d = boots[1][k] - boots[0][k]
            d = d[np.isfinite(d)]
            if d.size:
                lo, hi = np.nanpercentile(d, 5), np.nanpercentile(d, 95)
                delta = bases[1].get(k, np.nan) - bases[0].get(k, np.nan)
                sig = ((lo > 0) or (hi < 0)) and np.isfinite(delta)
                if sig and lower_better is not None:
                    mark = "better" if (delta < 0) == lower_better else "worse"
                    findings.append((label, delta, unit, mark, k))
                elif sig:
                    mark = "changed"
                    findings.append((label, delta, unit, mark, k))
                else:
                    mark = "—"
                if not np.isfinite(delta):
                    row += " not estimable in both flights | — |"
                else:
                    row += f" {delta:+.3g} [{lo:+.3g}–{hi:+.3g}] | {mark} |"
            else:
                row += " – | – |"
        elif len(names) == 2:
            d = bases[1].get(k, np.nan) - bases[0].get(k, np.nan)
            row += f" {d:+.3g} (no CI) | — |"
        L.append(row)
    L.append("")

    if len(names) == 2:
        mm = [b.get("mean_motor", np.nan) for b in bases]
        if np.isfinite(mm).all() and abs(mm[1] - mm[0]) / max(mm[0], 1e-6) > 0.05:
            L.append(f"> **Comparability:** the mean motor command changed by "
                     f"{100*(mm[1]-mm[0])/mm[0]:+.0f}% ({mm[0]:.3f} -> {mm[1]:.3f}). "
                     f"That is weight, battery health or propulsion efficiency — the "
                     f"two flights are not in the same condition, and no performance "
                     f"difference below should be attributed to configuration "
                     f"alone.\n")
        L.append("### Reading\n")
        exc_changed = any(k == "exc_p" for *_, k in findings)
        robust = [d for d in findings if d[4] in ROBUST]
        sensitive = [d for d in findings if d[4] in CONDITION_SENSITIVE]
        context = [d for d in findings if d[4] not in ROBUST | CONDITION_SENSITIVE]
        if not findings:
            L.append("- no metric changed significantly: the differences fit inside "
                     "the internal variability of each flight")
        if robust:
            L.append("**Robust evidence** (metrics that do not scale with how hard "
                     "the aircraft was flown):\n")
            for label, delta, unit, mark, _ in robust:
                L.append(f"- **{label}**: {delta:+.3g} {unit} — {mark}")
            L.append("")
        if sensitive:
            L.append("**Depends on conditions** (grows with how hard the flight "
                     "was):\n")
            for label, delta, unit, mark, _ in sensitive:
                L.append(f"- {label}: {delta:+.3g} {unit} — {mark}")
            L.append("")
        if context:
            L.append("**Context**:\n")
            for label, delta, unit, mark, _ in context:
                L.append(f"- {label}: {delta:+.3g} {unit} — {mark}")
            L.append("")
        if exc_changed and sensitive:
            L.append("> Flight aggressiveness changed significantly. The metrics in "
                     "the 'depends on conditions' list are contaminated by that: a "
                     "harder flight produces more error with no configuration change "
                     "at all. Use the robust list as evidence.")
        else:
            L.append("> Significant is not the same as caused by the configuration "
                     "change: wind, payload and piloting also change between "
                     "flights.")
        L.append("")

    # ------------------------------------------------------------ figure
    keys = [k for k in ("err_roll", "err_pitch", "err_p", "err_q",
                        "err_p_norm", "err_q_norm", "lag_p", "lag_q",
                        "K_p", "K_q", "band_p",
                        "vib_hf", "sat", "mean_motor", "exc_p")
            if any(np.isfinite(b.get(k, np.nan)) for b in bases)]
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow),
                            layout="constrained")
    fig.patch.set_facecolor(BG)
    axs = np.atleast_1d(axs).ravel()
    for ax, k in zip(axs, keys):
        label, unit, _ = METRICS[k]
        vals = [b.get(k, np.nan) for b in bases]
        los = [np.nanpercentile(s[k], 5) if np.isfinite(s[k]).any() else np.nan
               for s in boots]
        his = [np.nanpercentile(s[k], 95) if np.isfinite(s[k]).any() else np.nan
               for s in boots]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=[C[i % len(C)] for i in range(len(vals))], width=0.6)
        if k not in NO_CI:
            err = np.vstack([np.array(vals) - np.array(los),
                             np.array(his) - np.array(vals)])
            ax.errorbar(x, vals, yerr=np.abs(err), fmt="none", ecolor=INK,
                        capsize=4, lw=1)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=12)
        ax.set_title(f"{label} [{unit}]", fontsize=8.5, color=INK, loc="left")
        ax.set_facecolor(BG); ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=7)
        for s in ax.spines.values():
            s.set_color(GRID)
    for ax in axs[len(keys):]:
        ax.axis("off")
    fig.suptitle("Flight comparison — bars with 90 % bootstrap interval",
                 fontsize=11, color=INK, x=0.01, ha="left")

    os.makedirs(args.out, exist_ok=True)
    tag = "_".join(re.sub(r"[^0-9A-Za-z._+-]+", "", n) for n in names)[:60]
    png = os.path.join(args.out, f"comparison_{tag}.png")
    md = os.path.join(args.out, f"comparison_{tag}.md")
    fig.savefig(png, dpi=130)
    L.append(f"![comparison]({os.path.basename(png)})")
    open(md, "w").write("\n".join(L) + "\n")
    print(f"report: {md}")
    print(f"figure: {png}")
    for label, delta, unit, mark, k in findings:
        tag = ("robust" if k in ROBUST else
               "condition" if k in CONDITION_SENSITIVE else "context")
        print(f"  [{tag:9}] {mark:8} {label}: {delta:+.3g} {unit}")
    if len(names) == 2 and not findings:
        print("  no significant difference")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

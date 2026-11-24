"""Reading PX4 .ulg logs into pandas DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyulog import ULog

NAV_STATE = {
    0: "Manual", 1: "Altitude", 2: "Position", 3: "Mission", 4: "Hold", 5: "Return",
    6: "Acro", 7: "Offboard", 8: "Stabilized", 10: "Takeoff", 11: "Land",
    12: "Descend", 13: "FollowMe", 14: "PrecLand", 15: "Orbit",
}


def load(path: str, topics: list[str] | None = None) -> ULog:
    return ULog(path, topics)


def to_df(ulog: ULog, name: str, multi_id: int = 0) -> pd.DataFrame:
    """One topic as a DataFrame, indexed by seconds since the start of the log."""
    d = ulog.get_dataset(name, multi_id)
    df = pd.DataFrame({k: v for k, v in d.data.items()})
    t0 = ulog.start_timestamp
    df["t"] = (df["timestamp"].astype(np.float64) - t0) * 1e-6
    return df.set_index("t").sort_index()


def quat_to_euler(q0, q1, q2, q3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PX4 quaternion (w,x,y,z, NED/FRD) -> roll, pitch, yaw in radians."""
    q0, q1, q2, q3 = (np.asarray(x, dtype=np.float64) for x in (q0, q1, q2, q3))
    n = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    n[n == 0] = 1.0
    q0, q1, q2, q3 = q0 / n, q1 / n, q2 / n, q3 / n
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    sinp = np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return roll, pitch, yaw


def euler_df(ulog: ULog, topic: str, prefix: str, qname: str = "q") -> pd.DataFrame:
    df = to_df(ulog, topic)
    r, p, y = quat_to_euler(*[df[f"{qname}[{i}]"].to_numpy() for i in range(4)])
    return pd.DataFrame({f"{prefix}_roll": r, f"{prefix}_pitch": p, f"{prefix}_yaw": y},
                        index=df.index)


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def resample_join(frames: list[pd.DataFrame], hz: float,
                  t_start: float | None = None, t_end: float | None = None) -> pd.DataFrame:
    """Resample several DataFrames onto a uniform grid (linear interpolation)."""
    lo = max(f.index[0] for f in frames) if t_start is None else t_start
    hi = min(f.index[-1] for f in frames) if t_end is None else t_end
    grid = np.arange(lo, hi, 1.0 / hz)
    out = []
    for f in frames:
        f = f[~f.index.duplicated(keep="first")]
        num = f.select_dtypes(include=[np.number])
        res = pd.DataFrame(
            {c: np.interp(grid, num.index.to_numpy(), num[c].to_numpy()) for c in num.columns},
            index=grid)
        out.append(res)
    df = pd.concat(out, axis=1)
    df.index.name = "t"
    return df


def armed_intervals(ulog: ULog) -> list[tuple[float, float]]:
    """Armed intervals, in seconds since the start of the log."""
    try:
        vs = to_df(ulog, "vehicle_status")
    except (KeyError, IndexError):
        return []
    armed = vs["arming_state"].to_numpy() == 2  # ARMING_STATE_ARMED
    t = vs.index.to_numpy()
    out, start = [], None
    for ti, a in zip(t, armed):
        if a and start is None:
            start = ti
        elif not a and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def mode_intervals(ulog: ULog) -> list[tuple[float, float, str]]:
    try:
        vs = to_df(ulog, "vehicle_status")
    except (KeyError, IndexError):
        return []
    t = vs.index.to_numpy()
    ns = vs["nav_state"].to_numpy()
    out = []
    i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[i]:
            j += 1
        end = t[j + 1] if j + 1 < len(t) else t[-1]
        out.append((float(t[i]), float(end), NAV_STATE.get(int(ns[i]), str(int(ns[i])))))
        i = j + 1
    return out

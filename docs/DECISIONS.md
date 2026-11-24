# Engineering decision log

Dated record of the choices made while building this toolkit, including the ones
that turned out to be wrong. Numbers come from the public PX4 logs listed in
[`data/README.md`](../data/README.md) and can be reproduced with the scripts in
`src/`.

---

## 2026-11-20 — Environment and log selection

- Python 3.14 in `.venv/` with pyulog, numpy, pandas, matplotlib, scipy, pysindy.
- Layout: `src/` (code), `data/ulg/<log_id>/` (downloaded logs, out of git),
  `out/` (figures and reports), `docs/` (what ships with the repo).
- Log selection is scripted, not manual: `https://review.px4.io/dbinfo` returns a
  gzipped JSON index of **493,010 public flights**. `src/select_logs.py` filters it
  by vehicle type, duration, logged errors and time spent in manual/acro modes.
- Attitude setpoint is read from `q_d`. The older `roll_body/pitch_body/yaw_body`
  fields only exist on legacy firmware.
- Plot convention: measured `#1F6FB2`, commanded `#D1670B` dashed — a pair checked
  for colour-vision deficiency.

## 2026-11-20 — First dynamics identification (SINDy on angular rates)

Target: `p_dot, q_dot, r_dot` on log `22e75c8c` (quadrotor, v1.17), restricted to
**Manual** segments (240 s usable), 50 Hz grid, 10 Hz low-pass, Savitzky-Golay
derivative over a 100 ms window.

- **Input is `vehicle_torque_setpoint`** (normalised, 48 Hz, time-aligned with the
  gyro), not `actuator_motors` (only 10 Hz in this log). Swapping the input barely
  moved R² (0.71 vs 0.71 on roll), so motor logging rate was never the bottleneck.
- When torques are reconstructed from `actuator_motors`, they use the **real
  geometry recorded in the log** (`CA_ROTOR*_PX/PY/CT/KM`), not an assumed mixer.
  Side effect worth reporting to an operator: on this quad the arms are stored as
  ±1 (geometry never calibrated); on the ARK_FPV log they are ±0.1 m.
- **Two terms changed everything** (mean R² 0.65 → 0.92):
  1. a first-order lag on the command — a sweep picked **60 ms** (motor + ESC);
  2. `tau*T` terms — control effectiveness grows with thrust.
- STLSQ with a **relative** threshold (a fraction of the dominant term). An absolute
  threshold does not work because the scale differs per axis. Cross-checked against
  `pysindy.STLSQ`: agreement to < 1e-15.
- Model selection: alternating 10 s train/test blocks; among models within 3 % of the
  best test R², the one with fewest terms wins.
- Result: `p_dot` R² = 0.92 with **3 terms**; `q_dot` R² = 0.90 with 7; `r_dot`
  R² = 0.43 (yaw was barely excited, std 16 °/s, derivative dominated by noise).
- Limits recorded at the time: gyroscopic coefficients (`q*r`, `p*r`) were zeroed by
  the regression; damping came out positive (non-physical) on some axes. In free
  simulation the error grows roughly linearly with horizon (roll: 19 °/s at 0.25 s →
  118 °/s at 2 s). Good for derivative and short horizon, not a simulator.

## 2026-11-20 — Physics-constrained library: the inertias are not in this flight

`src/sindy_physical.py` couples the three Euler equations through a single pair
`ky = Iy/Ix`, `kz = Iz/Ix` (with `Ix = 1`), control effectiveness `(b0 + b1*T)*tau`
and damping `-c*omega`, all sign-constrained. Given `(ky, kz)` the rest is linear, so
the fit becomes a 2-D search and the whole cost surface can be swept.

**Main result: the inertia ratios are not identifiable from this flight.** Four
independent pieces of evidence:

1. the unconstrained optimum lands at `Iz/Ix = 0.30`, physically impossible for a
   flat quad (`Iz ≈ Ix + Iy`, so ≈ 2); forcing the correct value costs only +5.7 % SSE;
2. fitted per segment, `Iz/Ix` pins to the lower bound (0.30) in 6 of 8 segments and
   to the upper bound (4.0) in 1; with `kz` fixed at 2, the best `Iy/Ix` wanders from
   0.5 to 2.0 between segments;
3. the gyroscopic term contributes 9–30 % of the residual (22, 18 and 9 °/s² against
   residuals of 207, 221 and 29 °/s²);
4. **decisive test**: adding the gyroscopic term with the physically correct inertia
   *lowers* test R² (`q_dot`: 0.721 → 0.701).

Required excitation: `std(q·r)` would have to be ~27× larger (and `p·r` ~37×) for the
gyroscopic term to clear the residual. That needs fast roll/pitch **and** yaw at the
same time, which ordinary piloting never produces.

Cost of constraining the library: test R² drops from 0.94/0.90/0.43 (free) to
0.87/0.73/0.23. The gap is real unmodelled dynamics, not noise.

Consequence for the tool: **report identifiability** (cost surface plus the missing
excitation factor) instead of returning an inertia number with no caveat.

## 2026-11-20 — Searching the database: does any public log have enough excitation?

- Swept all 493,010 flights for time in Acro (688 logs with ≥ 20 s) and for test
  keywords (sysid, chirp, doublet, sweep, autotune — 121 logs); downloaded 33
  candidates (1.5 GB) and measured `std(q·r)`, `std(p·r)`, `std(p·q)` directly with
  `src/excitation_scan.py`.
- **The flight-mode label is not a proxy for excitation.** The logs with the most time
  in "Acro" (coaxial octos, industrial hexas, 800–2400 s) have `std(p)` of 3–7 °/s —
  they sit still in nav_state 6. The most excited log in the first round was flown in
  **Manual**, not Acro.
- One real find: **`ac5a5608`** (HolyBro QAV250, v1.17, 49 s armed in Acro) reaches
  `q = 714 °/s` with `r = −483 °/s` simultaneously, `std(q·r) = 5.26` — 12× the previous
  best, 0.35× the target of 15. It is the only log with non-zero coupling time
  (|q| and |r| both above 100 °/s: 0.86 %).
- On that log the gyroscopic term finally clears the noise in roll (174 °/s² against a
  122 °/s² residual). **But the whole model collapses**: test R² 0.06 / 0.03 / 0.02,
  residual 500–900 °/s². Causes checked: 33 % of armed time with a motor on the floor or
  ceiling (commanded torque is not delivered), and dynamics at 700 °/s outside the reach
  of rigid body plus first-order lag. Resampling from the raw 220 Hz gyro instead of the
  48 Hz `vehicle_angular_velocity` makes it **worse**, so this is model structure, not
  aliasing.
- **The two requirements fight each other.** Gyroscopic excitation demands violent
  manoeuvring; violent manoeuvring saturates the allocator and leaves the regime where
  the model holds. None of the 33 logs sits in the useful corner (excitation ≥ 30 % of
  target **and** saturation < 30 %).

## 2026-11-20 — Per-flight report

`src/flight_report.py` emits Markdown plus a panel figure with a verdict per area
(OK / ATENÇÃO / FALHA / SEM DADO) and an overall verdict: attitude tracking, vibration,
estimator, actuation, battery, log quality.

Interpretation choices that prevent false alarms:

- attitude error is judged **only in automatic modes** (Position/Mission/Hold/...); in
  Manual/Acro the setpoint is a stick step and a large RMS is expected, so it is
  reported as information, not as a fault;
- a **gravity** innovation ratio never fails a flight: the EKF rejects gravity fusion
  under acceleration by design;
- an `accel_vibration_metric` of zero is treated as "not populated by this firmware"
  (it happens on the v1.24 vendor fork), not as zero vibration; the high-pass accel RMS
  and the clipping counter are used instead.

## 2026-11-20 — Flight-to-flight comparison

`src/compare_flights.py` has three parts: a diff of the PX4 parameters stored in the
logs, metrics measured **only in automatic modes**, and uncertainty from a block
bootstrap (10 s blocks, 90 % CI). A difference is only called significant when the CI
of the difference excludes zero.

- Pair found in the database by looking for a repeated `vehicle_uuid` on firmware
  ≥ 1.14: HKUST_NXT_DUAL, two flights on the same day, same airframe, with
  `MC_ROLLRATE_P` 0.11 → 0.05 and `MC_PITCHRATE_P` 0.19 → 0.12. It is the only genuine
  tuning change found; most vehicles with several public flights never change a gain.
- **Result, and the methodological lesson**: with the gain halved, raw RMS errors
  doubled (roll rate 7.2 → 14.6 °/s) — but flight B was also flown twice as hard. The
  error **normalised by command did not move** (0.66 → 0.61, inside the CI). What moved
  cleanly was the **tracking lag: 21.2 → 47.6 ms** (+26.5 ms, CI [+14.7, +52.9]), which
  does not scale with aggressiveness. The loop got slower, exactly as theory predicts, and
  raw RMS would have given the right answer for the wrong reason.
- Hence metrics are split into **robust** (normalised error, lag, plant effectiveness)
  and **condition-sensitive** (raw RMS, saturation), with a warning when aggressiveness
  changed too.
- Guards added after watching metrics lie:
  - normalised error is only computed when `std(command) ≥ 3 °/s`;
  - lag is only reported when peak correlation is ≥ 0.5;
  - **lag is estimated inside each contiguous 10 s segment and the median taken.** The
    first version computed it over the whole window and then bootstrapped by resampling
    shuffled blocks, which splices the signal: the point estimate (75.4 ms) landed above
    the upper bound of its own interval, because estimate and interval were different
    statistics. Corrected, the roll lag reads 21.2 → 47.6 ms instead of 21.8 → 75.4 ms,
    and the conclusion (the loop got significantly slower) is unchanged;
  - a comparability warning fires when mean motor command differs by more than 5 %
    between flights — that is weight, battery or propulsion, and no performance
    difference can then be attributed to configuration alone.
- Validation: splitting one flight in half (`--split`) flags no raw metric, only mean
  motor command (+0.029), which is the battery sagging through the flight.

## 2026-11-23 — Four standard methods that had been skipped

The earlier negative result rested only on time-domain sparse regression with a
numerical derivative, fitted per flight, with no closed-loop treatment. Four standard
methods attack exactly those weaknesses.

**1. Torque-free rotation — not present in these logs.** Detector: all motors ≤ 0.05
with |ω| ≥ 30 °/s. Across six logs it appears once, for 0.71 s, **after disarm with the
aircraft on the ground**. In the crash log the 561–946 °/s peaks all happen under
commanded torque. The idea is sound; the data is not there.

**2. Weak (integral) form SINDy (`src/sindy_weak.py`) — confirms the negative result and
cleans it up.** Integration by parts against a `(1−τ²)^6` test function, no numerical
derivative anywhere. Test R², under the same model-selection rule as the strong form
(fewest terms within 3 % of the best): 0.902 / 0.848 / 0.309 against 0.924 / 0.896 /
0.428 — equivalent, slightly worse. `Iz/Ix` still pins to 0.30 and ranges 0.30–4.00
across segments. The decisive test moves from **harmful** (strong form, `q_dot` −0.021)
to **indifferent** (+0.002): without derivative noise the gyroscopic term is not
damaging, it simply carries no information. With `kz` fixed at 2, the best `Iy/Ix`
ranges 0.6–2.0 across segments, so the apparent `Iy/Ix = 1.01 ± 0.08` from the joint fit
was a parameterisation artefact.

**3. Instrumental variables / 2SLS (`src/sindy_iv.py`) — strong instrument, small
correction.** Instrument: `manual_control_setpoint` (pilot stick) at lags 0 to 0.5 s.
Although logged at only 4.9 Hz, the first stage gives R² 0.73 (τx), 0.75 (τy), 0.95 (T)
— a **strong** instrument; only τz is weak (0.27). The non-physical damping survives
(−0.64 → −0.51 1/s). `b0` and `b1` individually explode (111 → 342 and 85 → −367), but
the **effective gain `b0 + b1·T̄` is stable** (152 → 164, +7 % in roll; +1 % in pitch).
So `tau` and `tau*T` are not separable, the effective gain is, and IV does not rescue
the damping.

**4. Frequency domain with coherence (`src/freq_id.py`) — the most useful new result.**
Welch with 8 s segments. **Coherence of 0.99 between 0.25 and 8 Hz** in roll and pitch:
in that band the torque→rate relation is excellent, something no time-domain metric
showed. The measured `|H|` **peaks at 0.5–0.75 Hz and falls off below that** — the direct
visual signature of the closed loop suppressing low-frequency error.

Structure comparison, weighted by coherence, on the roll axis:

| structure | parameters | R² |
|---|---|---|
| `K e^{-s·td}/(s+a)`, free pole | a = −0.91 1/s, td = 40 ms, K = 106 | 0.869 |
| `K/(s(1+s·Tm))` integrator + motor | Tm = 60 ms, K = 144 | 0.921 |
| `K e^{-s·td}/s` pure integrator | td = 50 ms, K = 129 | 0.916 |
| `K e^{-s·td}/(s(1+s·Tm))` | Tm = 40 ms, td = 20 ms, K = 138 | **0.937** |

**The negative damping was the wrong structure, not bias.** Roll is well described as a
pure integrator plus ~60 ms actuator lag with no aerodynamic damping — which is what
physics predicts for a small multirotor at these rates. With the right structure the
frequency-domain effectiveness (138–144) meets the time-domain one (152), narrowing a
~50 % discrepancy to ~10 %. On pitch the comparison is inconclusive (the free pole still
wins, 0.73 against 0.63).

**Corrected claim.** Not "inertia ratios cannot be identified from operational logs",
but: *they cannot be identified from these operational logs by time-domain sparse
regression, nor in weak form, nor with instrumental variables — and the frequency
response shows why: below ~0.5 Hz the closed loop dominates, and above it the
gyroscopic term sits under the residual.* What is identifiable with confidence is
control effectiveness (to ±50 %, or ±10 % with the right structure), the coherent
bandwidth, and the actuation delay (30–60 ms depending on method).

## 2026-11-24 — Plant effectiveness: a number that should not have been printed

Reviewing the comparison figure raised a fair question: the frequency-domain plant
effectiveness in roll read 510 on one flight and 384 on the other — a 25 % change on
the **same aircraft**, printed with no interval, and listed among the metrics that are
supposed to be robust to how the flight was flown.

Checked, in order:

- **Configuration.** The allocation parameters (`CA_ROTOR*_PX/PY/CT/KM`), `THR_MDL_FAC`
  and the battery scaling parameters are byte-identical between the two logs, so the
  mapping from normalised torque to physical torque did not change. Ruled out.
- **Operating point.** Mean motor command differs by 4 %, nowhere near enough.
- **The estimate itself.** This was it. The coherent set on the first flight is
  **17 frequency bins**; on the second, 41. Refitting both over a common band leaves the
  first flight with fewer than five usable points, and swapping in the better plant
  structure (integrator plus delay) makes its estimate jump from 6 to 780 — the sign of
  a fit with nothing to hold on to.

So the aircraft never changed; the estimate on the quieter flight was unsupported and
my code printed it anyway. The first flight was flown half as hard (roll-rate command
std 11 vs 24 °/s), which excites the plant less, drops coherence, and makes K
unreliable — **K does not depend on how hard you fly, but its reliability does.**

Fixed in `src/compare_flights.py`:

- a quality gate: K is only reported with at least 25 coherent bins spanning at least a
  4:1 frequency ratio, and reads "not estimable" otherwise;
- an interval, by resampling the coherent bins. Pitch now reads 153 [151–161] against
  166 [159–176] — overlapping, so the +8 % that looked like a finding is not one;
- the number of coherent bins is reported as its own row, so a missing K comes with the
  reason next to it.

`band_p` (coherent bandwidth) moved out of the robust list: it is a property of the
measurement, not of the aircraft.

## 2026-11-23 — Threshold calibration against the public fleet

`src/calibrate_thresholds.py` downloads a sample of public logs (one per
`vehicle_uuid`, 90–260 s, firmware ≥ 1.13), measures them with **the same functions the
report uses**, deletes each log immediately (disk stays flat) and builds the empirical
distribution. Run over 220 ids → **209 logs measured**, 11 download failures.

**The main return was not the thresholds: it was three of my own metrics breaking on
real data.** Six hand-picked logs show none of this; 209 random ones show it in minutes.

| metric | fleet p99 before the fix | diagnosis |
|---|---|---|
| `innov_max` | 6.3 × 10¹² | innovation ratio of an uninitialised EKF — numeric garbage, not "large innovation" |
| `vcel_min` | 47.9 V/cell (p10 = 0.025) | `cell_count` zero or wrong; divided by it without checking |
| `desbal_pct` | 6.1 × 10⁶ | division by a near-zero mean command with the aircraft on the ground |

Fixes: ratios above 1e3 are discarded and reported as "filter diverged or not
initialised"; cell voltage is validated against the physical 2.5–4.35 V range and the
cell count is **inferred** when the field is wrong (after the fix: n = 175, range
3.22–4.18 V/cell); imbalance requires a mean command above 0.05 and is discarded above
300 % as a broken measurement (1 log in 201).

Calibration rule: **ATENÇÃO = fleet p75, FALHA = fleet p90**. The first attempt used
p90/p99 and produced a failure threshold of **37.5°** of attitude error — calibrating
against a bad fleet that way only legitimises the bad.

Two deliberate exceptions, documented in the code:

- innovation ratio stays at 0.5 / 1.0 (fleet: 6.9 / 47.9) because a ratio above 1 means
  the EKF **rejected** the measurement, and a poor fleet does not make rejection
  acceptable;
- dropouts stay at 0.5 / 2.0 s (fleet p75 is 0.024 s, far too tight to be a warning).

Better than the threshold itself, the report now carries a **"position in the public
fleet"** section placing every metric in the measured distribution, e.g. "accelerometer
clipping: 4989, 96th percentile (n = 208)". That is checkable and does not depend on
guessing the right cut.

Standing limitation: the reference is the *public* fleet, biased toward people testing
new hardware. For use with an operator, the right calibration is over that operator's
own fleet — the script takes any list of log ids.

# data/

Public logs downloaded from [PX4 Flight Review](https://review.px4.io) /
`cdn.logs.px4.io`. Untrusted content: not versioned, never executed from here.
Each log lives in `ulg/<log_id>/<log_id>.ulg`.

| log_id | date | duration | vehicle | firmware | why it was picked |
|---|---|---|---|---|---|
| 22e75c8c-2fc0-4719-8af2-d87188faead8 | 2026-11-16 | 612 s | Quadrotor, PX4_FMU_V6C | v1.17.0 | 473 s in Manual, aggressive manoeuvring (rates to 256 °/s), zero dropouts — main working log |
| 1572eefa-5344-484a-96d6-6b1bac1ea123 | 2026-11-12 | 424 s | Quadrotor FPV, ARK_FPV | v1.18.0 | 1 kHz logging of rates and `actuator_motors`, a segment in Acro |
| 7a74d699-64ce-4c27-9783-8a4ca7add105 | 2026-11-14 | 1094 s | Hexarotor, GDCX_H743VI | v1.24 (vendor fork) | autonomous flight (584 s in Mission) as a contrast case |
| ac5a5608-5fc8-4422-9f2d-3ae25ab7c2c3 | 2026-11 | 186 s | Quadrotor, HolyBro QAV250 | v1.17.0 | highest gyroscopic excitation found in 493,010 flights (q = 714 °/s with r = −483 °/s); ends in loss of control |
| b267cdd1-5c9b-4332-9243-b97aa8e8bc7a | 2026-11-19 | 168 s | Quadrotor, HKUST_NXT_DUAL | v1.15.4 | comparison pair: `MC_ROLLRATE_P` = 0.11 |
| 69b8b2b4-8531-4651-9fc1-9c1261baf921 | 2026-11-19 | 200 s | same `sys_uuid` as above | v1.15.4 | same vehicle, same day, `MC_ROLLRATE_P` = 0.05 — the only real tuning change found in the public database |

To fetch one again:

```bash
curl -sSL -o data/ulg/<id>/<id>.ulg https://cdn.logs.px4.io/<id>.ulg
```

The full public index is at `https://review.px4.io/dbinfo` (gzipped JSON, ~390 MB
uncompressed, 493,010 flights). A further 39 logs were downloaded for the excitation
sweep and the threshold calibration, then deleted; their measurements are kept in
`out/excitation_scan.txt` and `docs/fleet_percentiles.json`.

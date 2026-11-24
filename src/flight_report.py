"""Relatorio automatico de um voo PX4 a partir do log .ulg.

Gera um Markdown com veredito por area (rastreamento de atitude, vibracao,
estimador, atuacao, bateria, qualidade do log) e uma figura com os painEis
correspondentes.

Os limiares sao escolhas explicitas, documentadas em LIMIARES abaixo, e nao
verdades universais: servem como ponto de partida para comparar voos do mesmo
veiculo. Um numero isolado fora do limiar nao e defeito; e motivo para olhar.

Uso: python src/flight_report.py data/ulg/<id>/<id>.ulg [--out out]
"""
from __future__ import annotations

import argparse
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
OK, WARN, FAIL, NA = "OK", "ATENCAO", "FALHA", "SEM DADO"
COLOR = {OK: "#3C7D4E", WARN: "#C98A14", FAIL: "#B3261E", NA: "#9A9A9A"}
C_A, C_B = "#1F6FB2", "#D1670B"
INK, MUTED, GRID, BG = "#1a1a1a", "#6b6b6b", "#e3e3e0", "#fcfcfb"

# modos em que o setpoint de atitude e gerado suavemente pelo proprio PX4;
# nos modos pilotados o erro grande e esperado (degrau de stick)
MODOS_AUTO = {"Position", "Hold", "Mission", "Return", "Takeoff", "Land",
              "Descend", "Orbit", "Offboard", "PrecLand", "FollowMe"}

# Limiares calibrados em 2026-11-23 contra 209 logs publicos da PX4 Flight
# Review (um por veiculo, 90-260 s, firmware >= 1.13): ATENCAO = p75 da frota,
# FALHA = p90. Distribuicoes em out/calibracao.json.
# Duas excecoes deliberadas, onde o significado fisico vence o percentil:
#   - innov_*: a frota tem p75 = 6,9 e p90 = 47,9, mas razao de inovacao > 1
#     significa medida REJEITADA pelo EKF. A frota ser ruim nao torna rejeicao
#     aceitavel, entao ficam 0,5 / 1,0.
#   - dropout_*: p75 da frota e 0,024 s, estreito demais para servir de aviso;
#     adotado 0,5 / 2,0 s (p90 da frota = 2,02 s).
LIMIARES = {
    "att_auto_warn_deg": 2.5, "att_auto_fail_deg": 6.0,      # frota: 2,35 / 6,02
    "rate_auto_warn_dps": 13.0, "rate_auto_fail_dps": 25.0,  # frota: 13,5 / 25,5
    "vib_warn": 20.0, "vib_fail": 33.0,                      # frota: 20,5 / 33,1
    "hf_accel_warn": 5.0, "hf_accel_fail": 8.0,              # frota: 4,84 / 7,93
    "innov_warn": 0.5, "innov_fail": 1.0,                    # excecao (ver acima)
    "sat_warn_pct": 7.0, "sat_fail_pct": 35.0,               # frota: 7,36 / 34,7
    "desbal_warn_pct": 30.0, "desbal_fail_pct": 54.0,        # frota: 30,5 / 53,8
    "cell_warn_v": 3.5, "cell_fail_v": 3.3,                  # frota: p25 3,5 / p10 3,22
    "dropout_warn_s": 0.5, "dropout_fail_s": 2.0,            # excecao (ver acima)
}


@dataclass
class Check:
    area: str
    status: str
    resumo: str
    linhas: list[str] = field(default_factory=list)


def pior(*st: str) -> str:
    for s in (FAIL, WARN, OK, NA):
        if s in st:
            return s
    return NA


def _armed_mask(t: np.ndarray, armed) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for a, b in armed:
        m |= (t >= a) & (t <= b)
    return m if m.any() else np.ones(len(t), dtype=bool)


# --------------------------------------------------------------- checagens
def check_atitude(u, ctx) -> tuple[Check, dict]:
    try:
        att = px4io.euler_df(u, "vehicle_attitude", "act")
        sp = px4io.euler_df(u, "vehicle_attitude_setpoint", "sp", qname="q_d")
        rate = px4io.to_df(u, "vehicle_angular_velocity")[["xyz[0]", "xyz[1]", "xyz[2]"]]
        rate.columns = ["act_p", "act_q", "act_r"]
        rsp = px4io.to_df(u, "vehicle_rates_setpoint")[["roll", "pitch", "yaw"]]
        rsp.columns = ["sp_p", "sp_q", "sp_r"]
    except Exception as e:
        return Check("Rastreamento de atitude", NA, f"topicos ausentes ({e})"), {}
    df = px4io.resample_join([att, sp, rate, rsp], 50.0)
    t = df.index.to_numpy()
    df = df[_armed_mask(t, ctx["armed"])]
    t = df.index.to_numpy()

    grupos = {"automatico": np.zeros(len(t), bool), "pilotado": np.zeros(len(t), bool)}
    for a, b, nm in ctx["modos"]:
        sel = (t >= a) & (t <= b)
        grupos["automatico" if nm in MODOS_AUTO else "pilotado"] |= sel

    out, linhas, st = {}, [], NA
    for g, m in grupos.items():
        if m.sum() < 50:
            continue
        e = {}
        for nome, sp_c, act_c, ang in [("roll", "sp_roll", "act_roll", True),
                                       ("pitch", "sp_pitch", "act_pitch", True),
                                       ("yaw", "sp_yaw", "act_yaw", True),
                                       ("p", "sp_p", "act_p", False),
                                       ("q", "sp_q", "act_q", False),
                                       ("r", "sp_r", "act_r", False)]:
            d = df[sp_c].to_numpy()[m] - df[act_c].to_numpy()[m]
            if ang:
                d = px4io.wrap_pi(d)
            e[nome] = float(np.sqrt(np.mean(d ** 2))) * R2D
        out[g] = {"n_s": float(m.sum()) / 50.0, **e}
        linhas.append(f"- **{g}** ({m.sum()/50:.0f} s): RMS roll {e['roll']:.1f}°, "
                      f"pitch {e['pitch']:.1f}°, yaw {e['yaw']:.1f}° | "
                      f"p {e['p']:.1f}, q {e['q']:.1f}, r {e['r']:.1f} °/s")
        if g == "automatico":
            att_max = max(e["roll"], e["pitch"])
            rate_max = max(e["p"], e["q"])
            s = OK
            if att_max > LIMIARES["att_auto_fail_deg"] or rate_max > LIMIARES["rate_auto_fail_dps"]:
                s = FAIL
            elif att_max > LIMIARES["att_auto_warn_deg"] or rate_max > LIMIARES["rate_auto_warn_dps"]:
                s = WARN
            st = pior(st, s)
    if "automatico" not in out:
        linhas.append("- sem tempo suficiente em modo automatico: o RMS em modo "
                      "pilotado nao e criterio de qualidade (degrau de stick)")
        st = NA
    resumo = (f"RMS auto roll/pitch = {out['automatico']['roll']:.1f}/"
              f"{out['automatico']['pitch']:.1f}°" if "automatico" in out
              else "so modos pilotados")
    return (Check("Rastreamento de atitude", st, resumo, linhas),
            {"df": df, "grupos": grupos, "rms": out})


def check_vibracao(u, ctx) -> tuple[Check, dict]:
    linhas, st, data = [], NA, {}
    try:
        sc = px4io.to_df(u, "sensor_combined")
        t = sc.index.to_numpy()
        m = _armed_mask(t, ctx["armed"])
        fs = 1.0 / np.median(np.diff(t[m]))
        b, a = butter(4, min(20.0 / (fs / 2), 0.95), btype="high")
        hf = [float(np.std(filtfilt(b, a, sc[f"accelerometer_m_s2[{i}]"].to_numpy()[m])))
              for i in range(3)]
        data["hf"] = hf
        worst = max(hf)
        s = (FAIL if worst > LIMIARES["hf_accel_fail"] else
             WARN if worst > LIMIARES["hf_accel_warn"] else OK)
        st = pior(st, s)
        linhas.append(f"- acelerometro acima de 20 Hz (RMS): "
                      f"x {hf[0]:.2f}, y {hf[1]:.2f}, z {hf[2]:.2f} m/s² "
                      f"(limiar {LIMIARES['hf_accel_warn']:.0f}/{LIMIARES['hf_accel_fail']:.0f})")
    except Exception as e:
        linhas.append(f"- sensor_combined indisponivel ({e})")

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
            # alguns firmwares nao preenchem a metrica: nao confundir com zero
            metrica_ok = np.isfinite(vmax) and vmax > 0.2
            data.setdefault("vib", []).append((mid, vmed, vmax, clip, metrica_ok))
            s = (FAIL if (metrica_ok and vmax > LIMIARES["vib_fail"]) or clip > 1000 else
                 WARN if (metrica_ok and vmax > LIMIARES["vib_warn"]) or clip > 0 else OK)
            st = pior(st, s)
            if metrica_ok:
                linhas.append(f"- IMU {mid}: metrica de vibracao mediana {vmed:.1f}, "
                              f"pico {vmax:.1f} m/s² (limiar {LIMIARES['vib_warn']:.0f}/"
                              f"{LIMIARES['vib_fail']:.0f}); clipping: {clip}")
            else:
                linhas.append(f"- IMU {mid}: metrica de vibracao nao preenchida por "
                              f"este firmware; clipping do acelerometro: {clip}")
    except Exception as e:
        linhas.append(f"- vehicle_imu_status indisponivel ({e})")
    if data.get("vib"):
        clip_tot = sum(v[3] for v in data["vib"])
        com_metrica = [v for v in data["vib"] if v[4]]
        resumo = (f"pico {max(v[2] for v in com_metrica):.0f} m/s²" if com_metrica
                  else f"HF {max(data.get('hf', [0])):.1f} m/s²")
        if clip_tot:
            resumo += f", {clip_tot} clipping"
    else:
        resumo = "parcial"
    return Check("Vibracao", st, resumo, linhas), data


def check_estimador(u, ctx) -> tuple[Check, dict]:
    linhas, st, data = [], NA, {}
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
            # razoes acima de 1e3 sao lixo numerico (filtro nao inicializado),
            # nao "inovacao muito grande": nao entram na estatistica
            divergiu = float(np.max(v)) > 1e3
            v = v[v <= 1e3]
            if v.size == 0:
                ratios[f"{c}"] = (np.nan, np.inf, 100.0)
                continue
            ratios[f"{c}"] = (float(np.nanmedian(v)), float(np.nanmax(v)),
                              100.0 * float(np.mean(v > 1.0)))
            if divergiu:
                linhas.append(f"- {c}: valores acima de 1e3 descartados "
                              f"(filtro divergido ou nao inicializado)")
        if ratios:
            data["serie"] = (t[m], d, [c for c in d.columns if c in ratios])
            break
    if ratios:
        st = OK
        for k, (med, mx, frac) in sorted(ratios.items(), key=lambda kv: -kv[1][1])[:8]:
            # a fusao de gravidade e rejeitada por projeto sob aceleracao: so
            # informa, nao reprova o voo
            info = k.startswith("gravity")
            s = OK if info else (FAIL if frac > 1.0 else
                                 WARN if mx > LIMIARES["innov_warn"] else OK)
            st = pior(st, s)
            linhas.append(f"- {k}: mediana {med:.2f}, pico {mx:.2f}, "
                          f"{frac:.2f}% do tempo acima de 1 (rejeicao)"
                          + ("  *(informativo: gravidade e rejeitada sob aceleracao)*"
                             if info else ""))
        data["ratios"] = ratios
        criticas = {k: v for k, v in ratios.items() if not k.startswith("gravity")}
        data["criticas"] = criticas
    else:
        linhas.append("- sem razoes de inovacao no log")

    try:
        es = px4io.to_df(u, "estimator_status")
        t = es.index.to_numpy(); m = _armed_mask(t, ctx["armed"])
        ff = es["filter_fault_flags"].to_numpy()[m]
        nf = int(np.count_nonzero(ff))
        if nf:
            st = pior(st, FAIL)
            linhas.append(f"- **filter_fault_flags nao zero em {nf} amostras**")
        for c in [c for c in es.columns if c.startswith("reset_count")]:
            v = es[c].to_numpy()[m]
            n = int(v[-1] - v[0]) if len(v) else 0
            if n > 0:
                st = pior(st, WARN if n < 5 else FAIL)
                linhas.append(f"- {c}: {n} resets durante o voo")
    except Exception:
        pass
    crit = data.get("criticas") or {}
    resumo = (f"pior razao (sem gravidade) {max(v[1] for v in crit.values()):.2f}"
              if crit else ("so gravidade" if ratios else "sem dado"))
    return Check("Estimador (EKF)", st, resumo, linhas), data


def check_atuacao(u, ctx) -> tuple[Check, dict]:
    linhas, st, data = [], NA, {}
    try:
        am = px4io.to_df(u, "actuator_motors")
        cols = [c for c in am.columns if c.startswith("control[")]
    except Exception:
        try:
            am = px4io.to_df(u, "actuator_outputs")
            cols = [c for c in am.columns if c.startswith("output[")]
        except Exception as e:
            return Check("Atuacao", NA, f"sem comandos de motor ({e})"), {}
    t = am.index.to_numpy()
    m = _armed_mask(t, ctx["armed"])
    U = np.column_stack([am[c].to_numpy() for c in cols])[m]
    with np.errstate(all="ignore"):
        cheia = ~np.all(np.isnan(U), axis=0)
        desvio = np.zeros(U.shape[1])
        if cheia.any():
            desvio[cheia] = np.nanstd(U[:, cheia], axis=0)
    keep = cheia & (desvio > 1e-6)
    U = U[:, keep]
    if U.size == 0:
        return Check("Atuacao", NA, "comandos constantes/NaN"), {}
    if np.nanmax(U) > 10:                      # PWM em us -> normaliza
        U = (U - 1000.0) / 1000.0
    sat = 100.0 * float(np.mean(np.any((U <= 0.02) | (U >= 0.95), axis=1)))
    med = np.nanmean(U, axis=0)
    # com comando medio quase nulo (veiculo no chao) a razao explode
    desbal = (100.0 * float((med.max() - med.min()) / med.mean())
              if float(med.mean()) > 0.05 else np.nan)
    data["U"] = U; data["med"] = med; data["t"] = t[m]
    s_sat = (FAIL if sat > LIMIARES["sat_fail_pct"] else
             WARN if sat > LIMIARES["sat_warn_pct"] else OK)
    s_des = (NA if not np.isfinite(desbal) else
             FAIL if desbal > LIMIARES["desbal_fail_pct"] else
             WARN if desbal > LIMIARES["desbal_warn_pct"] else OK)
    st = pior(s_sat, s_des)
    linhas.append(f"- saturacao do alocador: {sat:.1f}% do tempo armado com algum "
                  f"motor <= 0,02 ou >= 0,95 (limiar {LIMIARES['sat_warn_pct']:.0f}/"
                  f"{LIMIARES['sat_fail_pct']:.0f}%)")
    linhas.append("- comando medio por motor: " +
                  ", ".join(f"{v:.3f}" for v in med) +
                  f"  -> desbalanceamento {desbal:.1f}% "
                  if np.isfinite(desbal) else
                  "  -> desbalanceamento nao avaliado (empuxo medio < 0,05) "
                  f"(limiar {LIMIARES['desbal_warn_pct']:.0f}/"
                  f"{LIMIARES['desbal_fail_pct']:.0f}%)")
    if np.isfinite(desbal) and desbal > LIMIARES["desbal_warn_pct"]:
        linhas.append("  - desbalanceamento alto costuma ser CG deslocado, braco "
                      "torto, helice danificada ou motor fraco")
    linhas.append(f"- margem media de empuxo: {100*(1-np.nanmean(U)):.0f}% "
                  f"(comando medio {np.nanmean(U):.3f})")
    res_des = f"{desbal:.0f}%" if np.isfinite(desbal) else "n/d"
    return Check("Atuacao", st, f"saturacao {sat:.0f}%, desbalanceamento {res_des}",
                 linhas), data


def check_bateria(u, ctx) -> tuple[Check, dict]:
    try:
        bs = px4io.to_df(u, "battery_status")
    except Exception as e:
        return Check("Bateria", NA, f"sem battery_status ({e})"), {}
    t = bs.index.to_numpy(); m = _armed_mask(t, ctx["armed"])
    vcol = "voltage_filtered_v" if "voltage_filtered_v" in bs.columns else "voltage_v"
    if vcol not in bs.columns:
        return Check("Bateria", NA, "sem tensao no log"), {}
    v = bs[vcol].to_numpy()[m]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Check("Bateria", NA, "tensao invalida"), {}
    cells = bs["cell_count"].to_numpy()[m] if "cell_count" in bs.columns else np.array([0])
    nc = int(np.nanmax(cells)) if np.isfinite(np.nanmax(cells)) and np.nanmax(cells) > 0 else 0
    # cell_count errado/zerado e comum: valida contra faixa fisica de Li-ion
    def plausivel(n):
        return n > 0 and 2.5 <= float(np.nanmedian(v)) / n <= 4.35
    if not plausivel(nc):
        cand = [n for n in range(1, 15) if plausivel(n)]
        nc = cand[0] if cand else 0
    cur = bs["current_a"].to_numpy()[m] if "current_a" in bs.columns else np.array([np.nan])
    data = {"t": t[m], "v": v, "i": cur, "nc": nc}
    linhas, st = [], OK
    if nc:
        vc = v / nc
        vc = vc[(vc > 1.5) & (vc < 4.6)]
        if vc.size == 0:
            return Check("Bateria", NA, "tensao fora de faixa fisica"), {}
        linhas.append(f"- {nc}S: inicio {vc[0]:.2f} V/celula, fim {vc[-1]:.2f}, "
                      f"minimo {np.nanmin(vc):.2f} "
                      f"(limiar {LIMIARES['cell_warn_v']}/{LIMIARES['cell_fail_v']})")
        st = (FAIL if np.nanmin(vc) < LIMIARES["cell_fail_v"] else
              WARN if np.nanmin(vc) < LIMIARES["cell_warn_v"] else OK)
        sag = float(np.nanmax(vc) - np.nanmin(vc))
        linhas.append(f"- queda de tensao sob carga (sag): {sag:.2f} V/celula")
    else:
        linhas.append(f"- tensao: inicio {v[0]:.2f} V, fim {v[-1]:.2f}, "
                      f"minimo {np.nanmin(v):.2f} (contagem de celulas ausente)")
        st = NA
    if np.isfinite(cur).any():
        linhas.append(f"- corrente: media {np.nanmean(cur):.1f} A, pico {np.nanmax(cur):.1f} A")
    if "discharged_mah" in bs.columns:
        dm = bs["discharged_mah"].to_numpy()[m]
        linhas.append(f"- consumo: {np.nanmax(dm):.0f} mAh")
    if "remaining" in bs.columns:
        rem = bs["remaining"].to_numpy()[m]
        linhas.append(f"- restante estimado ao pousar: {100*np.nanmin(rem):.0f}%")
    return Check("Bateria", st,
                 f"min {np.nanmin(vc):.2f} V/cel" if nc else "sem celulas",
                 linhas), data


def check_log(u, ctx) -> tuple[Check, dict]:
    nd = len(u.dropouts)
    sd = sum(d.duration for d in u.dropouts) / 1000.0
    st = (FAIL if sd > LIMIARES["dropout_fail_s"] else
          WARN if sd > LIMIARES["dropout_warn_s"] else OK)
    linhas = [f"- dropouts: {nd} ({sd:.2f} s perdidos; limiar "
              f"{LIMIARES['dropout_warn_s']:.0f}/{LIMIARES['dropout_fail_s']:.0f} s)",
              f"- topicos gravados: {len(u.data_list)}"]
    msgs = [m for m in u.logged_messages if m.log_level <= 4]
    if msgs:
        st = pior(st, WARN)
        linhas.append(f"- {len(msgs)} mensagens de aviso/erro do firmware:")
        for m in msgs[:8]:
            linhas.append(f"  - t={m.timestamp*1e-6 - u.start_timestamp*1e-6:.1f}s: "
                          f"{m.message[:90]}")
    return Check("Qualidade do log", st, f"{nd} dropouts, {sd:.1f} s", linhas), {}


# ------------------------------------------------- numeros para calibracao
def medidas(u, d_att, d_vib, d_ekf, d_act, d_bat) -> dict:
    """Extrai, das mesmas checagens, os numeros que viram limiar.

    Usado pelo relatorio (para comparar com a frota) e por
    src/calibrate_thresholds.py (para montar a distribuicao da frota).
    """
    m: dict = {}
    a = (d_att or {}).get("rms", {}).get("automatico")
    if a:
        m.update(att_roll=a["roll"], att_pitch=a["pitch"], att_yaw=a["yaw"],
                 rate_p=a["p"], rate_q=a["q"], rate_r=a["r"], t_auto=a["n_s"])
    if (d_vib or {}).get("hf"):
        m["hf_accel"] = float(max(d_vib["hf"]))
    if (d_vib or {}).get("vib"):
        com = [v for v in d_vib["vib"] if v[4]]
        if com:
            m["vib_metric"] = float(max(v[2] for v in com))
        m["clipping"] = float(sum(v[3] for v in d_vib["vib"]))
    crit = (d_ekf or {}).get("criticas") or {}
    finitos = [v for v in crit.values() if np.isfinite(v[1])]
    if finitos:
        m["innov_max"] = float(max(v[1] for v in finitos))
        m["innov_rej_pct"] = float(max(v[2] for v in finitos))
    if (d_act or {}).get("med") is not None and len((d_act or {}).get("med", [])):
        U, med = d_act["U"], d_act["med"]
        m["sat_pct"] = 100.0 * float(np.mean(np.any((U <= 0.02) | (U >= 0.95), axis=1)))
        if float(np.nanmean(med)) > 0.05:
            dsb = 100.0 * float((med.max() - med.min()) / med.mean())
            if dsb <= 300.0:
                m["desbal_pct"] = dsb
        m["motor_medio"] = float(np.nanmean(U))
    if (d_bat or {}).get("nc"):
        vc = d_bat["v"] / d_bat["nc"]
        vc = vc[(vc > 1.5) & (vc < 4.6)]
        if vc.size:
            m["vcel_min"] = float(np.nanmin(vc))
    m["dropout_s"] = sum(d.duration for d in u.dropouts) / 1000.0
    return m


# metrica -> (rotulo, maior_e_pior)
FROTA_ROTULOS = {
    "att_roll": ("erro de atitude roll", True),
    "att_pitch": ("erro de atitude pitch", True),
    "rate_p": ("erro de taxa p", True),
    "rate_q": ("erro de taxa q", True),
    "hf_accel": ("vibracao acima de 20 Hz", True),
    "vib_metric": ("metrica de vibracao", True),
    "clipping": ("clipping do acelerometro", True),
    "innov_max": ("pior razao de inovacao", True),
    "sat_pct": ("saturacao do alocador", True),
    "desbal_pct": ("desbalanceamento de motores", True),
    "dropout_s": ("dropouts", True),
    "vcel_min": ("tensao minima por celula", False),
}


def percentis_frota(med: dict, caminho="out/calibracao_cru.json"):
    """Posiciona cada medida na distribuicao da frota publica medida."""
    if not os.path.exists(caminho):
        return None, 0
    try:
        cru = __import__("json").load(open(caminho))
    except Exception:
        return None, 0
    n = max((len(v) for v in cru.values()), default=0)
    out = []
    for k, (rot, maior_pior) in FROTA_ROTULOS.items():
        if k not in med or k not in cru:
            continue
        amostra = np.array([x for x in cru[k] if np.isfinite(x)])
        if amostra.size < 20:
            continue
        pct = 100.0 * float(np.mean(amostra <= med[k]))
        if not maior_pior:
            pct = 100.0 - pct
        out.append((rot, med[k], pct, amostra.size))
    return out, n


# ------------------------------------------------------------------ saida
def figura(path_png, nome, ctx, d_att, d_vib, d_ekf, d_act, d_bat, checks):
    fig = plt.figure(figsize=(14, 11), layout="constrained")
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.35, 1, 1, 1])

    # faixa de status
    axs = fig.add_subplot(gs[0, :]); axs.set_facecolor(BG); axs.axis("off")
    for i, c in enumerate(checks):
        x = i / len(checks)
        axs.add_patch(plt.Rectangle((x, 0.25), 1 / len(checks) - 0.01, 0.5,
                                    color=COLOR[c.status], alpha=0.85))
        axs.text(x + 0.006, 0.66, c.area[:24], fontsize=8, color="white", va="center")
        axs.text(x + 0.006, 0.48, c.status, fontsize=7.5, color="white", va="center",
                 weight="bold")
        axs.text(x + 0.006, 0.32, c.resumo[:30], fontsize=6.8, color="white",
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

    # atitude
    ax = fig.add_subplot(gs[1, 0]); style(ax, "erro de atitude", "deg")
    if d_att:
        df = d_att["df"]; t = df.index.to_numpy()
        for c, lab, col in [("roll", "roll", C_A), ("pitch", "pitch", C_B)]:
            e = px4io.wrap_pi(df[f"sp_{c}"].to_numpy() - df[f"act_{c}"].to_numpy()) * R2D
            ax.plot(t, e, lw=0.7, color=col, label=lab)
        auto = d_att["grupos"]["automatico"]
        if auto.any():
            ax.fill_between(t, -180, 180, where=auto, color="#1F6FB2", alpha=0.07,
                            label="modo automatico")
        ax.set_ylim(-60, 60); ax.legend(fontsize=7, frameon=False, ncol=3)

    # vibracao
    ax = fig.add_subplot(gs[1, 1]); style(ax, "vibracao (accel_vibration_metric)", "m/s²")
    try:
        for v in d_vib.get("vib", []):
            if not v[4]:
                continue
            s = px4io.to_df(ctx["u"], "vehicle_imu_status", v[0])
            ax.plot(s.index, s["accel_vibration_metric"], lw=0.9, label=f"IMU {v[0]}")
        if ax.get_legend_handles_labels()[0]:
            ax.axhline(LIMIARES["vib_warn"], color=COLOR[WARN], lw=1, ls="--")
            ax.axhline(LIMIARES["vib_fail"], color=COLOR[FAIL], lw=1, ls="--")
            ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
            if xlim:
                ax.set_xlim(*xlim)
        else:
            ax.text(0.5, 0.5, "metrica nao preenchida por este firmware",
                    transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
    except Exception:
        pass

    # ekf
    ax = fig.add_subplot(gs[2, 0]); style(ax, "razoes de teste de inovacao do EKF", "-")
    if d_ekf.get("serie"):
        tt, dd, cs = d_ekf["serie"]
        for c in cs[:6]:
            ax.plot(dd.index, dd[c], lw=0.8, label=c.replace("_test_ratio", ""))
        ax.axhline(1.0, color=COLOR[FAIL], lw=1, ls="--")
        if xlim:
            ax.set_xlim(*xlim)
            ax.relim(); ax.autoscale_view(scalex=False)
        ax.set_ylim(0, max(1.4, min(4.0, ax.get_ylim()[1])))
        ax.legend(fontsize=6.5, frameon=False, ncol=3)

    # atuacao
    ax = fig.add_subplot(gs[2, 1]); style(ax, "comando por motor", "comando [0..1]")
    if d_act:
        U, t = d_act["U"], d_act["t"]
        for i in range(U.shape[1]):
            ax.plot(t, U[:, i], lw=0.6, label=f"m{i}")
        ax.axhline(0.95, color=COLOR[WARN], lw=1, ls="--")
        ax.legend(fontsize=6.5, frameon=False, ncol=4)

    # bateria
    ax = fig.add_subplot(gs[3, 0]); style(ax, "bateria", "V/celula")
    if d_bat and d_bat.get("nc"):
        ax.plot(d_bat["t"], d_bat["v"] / d_bat["nc"], lw=1.1, color=C_A)
        ax.axhline(LIMIARES["cell_warn_v"], color=COLOR[WARN], lw=1, ls="--")
        ax.axhline(LIMIARES["cell_fail_v"], color=COLOR[FAIL], lw=1, ls="--")
        ax.set_xlabel("tempo [s]", fontsize=8, color=INK)

    # desbalanceamento
    ax = fig.add_subplot(gs[3, 1]); style(ax, "comando medio por motor",
                                          "comando medio [0..1]")
    if d_act:
        med = d_act["med"]
        ax.bar(np.arange(len(med)), med, color=C_A)
        ax.axhline(float(np.mean(med)), color=C_B, lw=1.2, ls="--", label="media")
        ax.set_xticks(np.arange(len(med)))
        ax.set_xticklabels([f"m{i}" for i in range(len(med))])
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"Relatorio de voo — {nome}", fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(path_png, dpi=120)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    u = px4io.load(args.ulg)
    nome = os.path.basename(args.ulg).replace(".ulg", "")
    ctx = {"u": u, "armed": px4io.armed_intervals(u), "modos": px4io.mode_intervals(u)}

    c_att, d_att = check_atitude(u, ctx)
    c_vib, d_vib = check_vibracao(u, ctx)
    c_ekf, d_ekf = check_estimador(u, ctx)
    c_act, d_act = check_atuacao(u, ctx)
    c_bat, d_bat = check_bateria(u, ctx)
    c_log, _ = check_log(u, ctx)
    checks = [c_att, c_vib, c_ekf, c_act, c_bat, c_log]

    dur = (u.last_timestamp - u.start_timestamp) * 1e-6
    t_arm = sum(b - a for a, b in ctx["armed"])
    modos = {}
    for a, b, nm in ctx["modos"]:
        modos[nm] = modos.get(nm, 0.0) + (b - a)

    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"relatorio_{nome[:8]}.png")
    md = os.path.join(args.out, f"relatorio_{nome[:8]}.md")
    figura(png, nome[:8], ctx, d_att, d_vib, d_ekf, d_act, d_bat, checks)

    L = []
    L.append(f"# Relatorio de voo — `{nome[:8]}`\n")
    L.append(f"- veiculo: {u.msg_info_dict.get('ver_hw','?')} "
             f"({u.msg_info_dict.get('ver_hw_subtype','')})")
    L.append(f"- firmware: {u.msg_info_dict.get('ver_sw_branch','?')} "
             f"`{str(u.msg_info_dict.get('ver_sw',''))[:10]}`")
    L.append(f"- duracao do log: {dur:.0f} s | armado: {t_arm:.0f} s")
    L.append("- tempo por modo: " + ", ".join(
        f"{k} {v:.0f} s" for k, v in sorted(modos.items(), key=lambda kv: -kv[1])
        if v >= 1))
    L.append("")
    pior_geral = pior(*[c.status for c in checks])
    L.append(f"**Veredito geral: {pior_geral}**\n")
    L.append("| area | status | resumo |")
    L.append("|---|---|---|")
    for c in checks:
        L.append(f"| {c.area} | {c.status} | {c.resumo} |")
    L.append("")
    for c in checks:
        L.append(f"## {c.area} — {c.status}\n")
        L.extend(c.linhas or ["- sem dados"])
        L.append("")
    med = medidas(u, d_att, d_vib, d_ekf, d_act, d_bat)
    frota, n_frota = percentis_frota(med)
    if frota:
        L.append("## Posicao na frota publica\n")
        L.append(f"Amostra de referencia: logs publicos da PX4 Flight Review, "
                 f"um por veiculo (n ate {n_frota}). O percentil diz quantos por "
                 f"cento da frota estao em situacao igual ou melhor que este voo.\n")
        L.append("| metrica | valor | percentil (100 = pior da frota) |")
        L.append("|---|---|---|")
        for rot, val, pct, nn in sorted(frota, key=lambda r: -r[2]):
            marca = (" **pior que a maioria**" if pct >= 90 else
                     " acima da mediana" if pct >= 50 else "")
            L.append(f"| {rot} | {val:.3g} | {pct:.0f}º (n={nn}){marca} |")
        L.append("")

    L.append("---")
    L.append("Limiares usados (escolha do autor, nao norma): " +
             ", ".join(f"`{k}`={v}" for k, v in LIMIARES.items()))
    L.append(f"\n![painel]({os.path.basename(png)})")
    open(md, "w").write("\n".join(L) + "\n")

    print(f"relatorio: {md}")
    print(f"figura:    {png}")
    print(f"\nveredito geral: {pior_geral}")
    for c in checks:
        print(f"  {c.status:8} {c.area:26} {c.resumo}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

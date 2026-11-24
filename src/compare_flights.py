"""Compara dois ou mais voos: o que mudou na configuracao e o que mudou no voo.

Responde a pergunta de ensaio: "mexi no ganho / troquei a helice / mudei o
firmware — melhorou?"

Tres partes:
  1. diff dos parametros PX4 gravados no log (o que de fato mudou entre os voos);
  2. metricas de desempenho medidas **so em modos automaticos**, onde o setpoint
     e gerado pelo proprio PX4 e a comparacao faz sentido;
  3. incerteza: cada metrica vem com intervalo de confianca por bootstrap de
     blocos de 10 s, e a diferenca entre voos so e chamada de significativa se o
     intervalo da diferenca nao contem zero.

Significativo nao quer dizer causado pela mudanca de parametro: vento, carga e
pilotagem tambem mudam entre voos. Por isso o relatorio imprime indicadores de
comparabilidade (agressividade do comando, empuxo medio, duracao util).

Uso:
  python src/compare_flights.py A.ulg B.ulg [C.ulg ...] [--nome rotulos]
  python src/compare_flights.py A.ulg --split 300     # valida: parte o voo em dois
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import freq_id as fq  # noqa: E402
from flight_report import MODOS_AUTO  # noqa: E402

R2D = 180.0 / np.pi
C = ["#1F6FB2", "#D1670B", "#5B8C3E", "#8E5AA8"]
INK, MUTED, GRID, BG = "#1a1a1a", "#6b6b6b", "#e3e3e0", "#fcfcfb"
HZ = 50.0
BLOCO_S = 10.0
NBOOT = 400
# metricas cujo valor depende de quanto o voo foi exigido (vento, missao, piloto)
SENSIVEIS = {"err_roll", "err_pitch", "err_p", "err_q", "sat"}
# metricas que nao escalam com a agressividade: evidencia mais limpa
ROBUSTAS = {"err_p_norm", "err_q_norm", "atraso_p", "atraso_q",
            "K_p", "K_q", "banda_p"}

PREFIX_TUNING = ("MC_", "MPC_", "ATT_", "IMU_", "CA_", "THR_", "MOT_", "PWM_",
                 "RATE_", "EKF2_", "SENS_")

# metrica -> (rotulo, unidade, menor_e_melhor)
METRICAS = {
    "err_roll": ("erro de atitude roll (RMS)", "deg", True),
    "err_pitch": ("erro de atitude pitch (RMS)", "deg", True),
    "err_p": ("erro de taxa p (RMS)", "deg/s", True),
    "err_q": ("erro de taxa q (RMS)", "deg/s", True),
    "err_p_norm": ("erro de p normalizado pelo comando", "-", True),
    "err_q_norm": ("erro de q normalizado pelo comando", "-", True),
    "atraso_p": ("atraso de rastreamento em p", "ms", True),
    "atraso_q": ("atraso de rastreamento em q", "ms", True),
    "vib_hf": ("vibracao acima de 20 Hz (pior eixo)", "m/s²", True),
    "sat": ("saturacao do alocador", "%", True),
    "motor_medio": ("comando medio de motor", "0..1", None),
    "exc_p": ("agressividade: desvio do comando de p", "deg/s", None),
    # propriedades de planta (frequencia): mudam com hardware, nao com ganho
    "K_p": ("efetividade de planta em p (frequencia)", "rad/s² por tau", None),
    "K_q": ("efetividade de planta em q (frequencia)", "rad/s² por tau", None),
    "banda_p": ("banda com coerencia >= 0.6 em p", "Hz", None),
}

SEM_IC = {"vib_hf", "K_p", "K_q", "banda_p"}


def janela_auto(u, t):
    m = np.zeros(len(t), dtype=bool)
    for a, b, nm in px4io.mode_intervals(u):
        if nm in MODOS_AUTO:
            m |= (t >= a) & (t <= b)
    arm = np.zeros(len(t), dtype=bool)
    for a, b in px4io.armed_intervals(u):
        arm |= (t >= a) & (t <= b)
    return m & arm


def atraso_ms(sp, act, hz, max_ms=400, corr_min=0.5):
    """Atraso que maximiza a correlacao cruzada entre comando e resposta.

    Devolve NaN quando a correlacao no pico e fraca (< corr_min): nesse caso o
    'atraso' seria so o ponto onde o ruido casou melhor, nao um atraso real.
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
    # interpolacao parabolica no pico: resolucao abaixo do passo de amostragem
    if 0 < k < len(cs) - 1:
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = (y0 - 2 * y1 + y2)
        k = k + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
    return float(k) / hz * 1000.0


def carregar(path):
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
    df = df[janela_auto(u, t)]
    # vibracao medida no proprio sensor_combined (nao reamostrado)
    vib = np.nan
    try:
        sc = px4io.to_df(u, "sensor_combined")
        ts = sc.index.to_numpy()
        ms = janela_auto(u, ts)
        if ms.sum() > 500:
            fs = 1.0 / np.median(np.diff(ts[ms]))
            b, a = butter(4, min(20.0 / (fs / 2), 0.95), btype="high")
            vib = max(float(np.std(filtfilt(b, a, sc[f"accelerometer_m_s2[{i}]"]
                                            .to_numpy()[ms]))) for i in range(3))
    except Exception:
        pass
    # resposta em frequencia: separa propriedade de planta de ajuste de malha
    freq = {}
    try:
        tq = px4io.to_df(u, "vehicle_torque_setpoint")[["xyz[0]", "xyz[1]"]]
        tq.columns = ["tau_x", "tau_y"]
        rt = px4io.to_df(u, "vehicle_angular_velocity")[["xyz[0]", "xyz[1]"]]
        rt.columns = ["pp", "qq"]
        d2 = px4io.resample_join([tq, rt], HZ)
        t2 = d2.index.to_numpy()
        d2 = d2[janela_auto(u, t2)]
        if len(d2) > 8 * HZ * 2:
            from scipy.signal import coherence as _coh, csd as _csd, welch as _welch
            nper = int(8 * HZ)
            for eixo, (cu, cy) in {"p": ("tau_x", "pp"), "q": ("tau_y", "qq")}.items():
                x = d2[cu].to_numpy(); y = d2[cy].to_numpy()
                f_, Pxx = _welch(x, fs=HZ, nperseg=nper)
                _, Pxy = _csd(x, y, fs=HZ, nperseg=nper)
                _, coh = _coh(x, y, fs=HZ, nperseg=nper)
                H = Pxy / np.where(Pxx == 0, np.nan, Pxx)
                b = (f_ > 0.05) & (f_ < 10) & (coh >= 0.6)
                if b.sum() >= 5:
                    melhor = None
                    for td in np.arange(0.0, 0.201, 0.01):
                        a_, K_, r_, _ = fq.ajusta_primeira_ordem(
                            f_[b], H[b] * np.exp(1j * 2 * np.pi * f_[b] * td), coh[b])
                        if melhor is None or r_ > melhor[2]:
                            melhor = (a_, K_, r_, td)
                    freq[f"K_{eixo}"] = abs(melhor[1])
                    if eixo == "p":
                        freq["banda_p"] = float(f_[b].max() - f_[b].min())
    except Exception:
        pass
    return u, df, mot, vib, freq


def metricas_bloco(df, mot, vib, idx, freq=None):
    """Metricas num subconjunto de amostras (um bloco ou um conjunto deles)."""
    d = df.iloc[idx]
    out = {}
    for nome, a, b in [("err_roll", "sp_roll", "act_roll"),
                       ("err_pitch", "sp_pitch", "act_pitch")]:
        e = px4io.wrap_pi(d[a].to_numpy() - d[b].to_numpy()) * R2D
        out[nome] = float(np.sqrt(np.mean(e ** 2)))
    for eixo in ("p", "q"):
        e = (d[f"sp_{eixo}"].to_numpy() - d[f"act_{eixo}"].to_numpy()) * R2D
        s = float(np.std(d[f"sp_{eixo}"].to_numpy()) * R2D)
        out[f"err_{eixo}"] = float(np.sqrt(np.mean(e ** 2)))
        # normalizar por um comando quase nulo infla a metrica: exige 3 deg/s
        out[f"err_{eixo}_norm"] = out[f"err_{eixo}"] / s if s >= 3.0 else np.nan
        out[f"atraso_{eixo}"] = atraso_ms(d[f"sp_{eixo}"].to_numpy(),
                                          d[f"act_{eixo}"].to_numpy(), HZ)
    out["exc_p"] = float(np.std(d["sp_p"].to_numpy()) * R2D)
    out["vib_hf"] = vib
    for k in ("K_p", "K_q", "banda_p"):
        out[k] = (freq or {}).get(k, np.nan)
    if mot:
        U = d[[f"m{i}" for i in range(len(mot))]].to_numpy()
        out["sat"] = 100.0 * float(np.mean(np.any((U <= 0.02) | (U >= 0.95), axis=1)))
        out["motor_medio"] = float(np.nanmean(U))
    else:
        out["sat"] = np.nan
        out["motor_medio"] = np.nan
    return out


def bootstrap(df, mot, vib, freq=None, nboot=NBOOT, rng=None):
    """Bootstrap de blocos: devolve (valor, amostras bootstrap) por metrica."""
    rng = rng or np.random.default_rng(0)
    n = len(df)
    nb = max(2, int(n / (BLOCO_S * HZ)))
    blocos = np.array_split(np.arange(n), nb)
    base = metricas_bloco(df, mot, vib, np.arange(n), freq)
    amostras = {k: [] for k in base}
    for _ in range(nboot):
        pick = rng.integers(0, len(blocos), len(blocos))
        idx = np.concatenate([blocos[i] for i in pick])
        m = metricas_bloco(df, mot, vib, idx, freq)
        for k, v in m.items():
            amostras[k].append(v)
    return base, {k: np.array(v, dtype=float) for k, v in amostras.items()}


def diff_parametros(ulogs, nomes):
    ps = [u.initial_parameters for u in ulogs]
    keys = set().union(*[set(p) for p in ps])
    ruido = ("CAL_", "COM_FLIGHT_UUID", "LND_FLIGHT_T", "RC1_", "RC2_", "RC3_",
             "RC4_", "RC5_", "RC6_", "RC7_", "RC8_", "RC9_", "RC1", "SENS_BOARD")
    tuning, outros = [], []
    for k in sorted(keys):
        vals = [p.get(k) for p in ps]
        if len({str(v) for v in vals}) == 1:
            continue
        (tuning if k.startswith(PREFIX_TUNING) else outros).append((k, vals))
    outros = [(k, v) for k, v in outros if not k.startswith(ruido)]
    return tuning, outros


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulgs", nargs="+")
    ap.add_argument("--nome", nargs="*", default=None)
    ap.add_argument("--split", type=float, default=None,
                    help="parte UM log em dois voos no instante dado (validacao)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    ulogs, dfs, mots, vibs, freqs, nomes = [], [], [], [], [], []
    if args.split is not None:
        u, df, mot, vib, freq = carregar(args.ulgs[0])
        base = os.path.basename(args.ulgs[0])[:8]
        for lab, sel in [(f"{base}[:{args.split:.0f}s]", df.index < args.split),
                         (f"{base}[{args.split:.0f}s:]", df.index >= args.split)]:
            ulogs.append(u); dfs.append(df[sel]); mots.append(mot); vibs.append(vib)
            freqs.append(freq); nomes.append(lab)
    else:
        for i, p in enumerate(args.ulgs):
            u, df, mot, vib, freq = carregar(p)
            ulogs.append(u); dfs.append(df); mots.append(mot); vibs.append(vib)
            freqs.append(freq)
            nomes.append(args.nome[i] if args.nome and i < len(args.nome)
                         else os.path.basename(p)[:8])

    for nome, df in zip(nomes, dfs):
        if len(df) < BLOCO_S * HZ * 2:
            print(f"ERRO: {nome} tem {len(df)/HZ:.0f} s em modo automatico, "
                  f"pouco para comparar")
            return 1

    rng = np.random.default_rng(42)
    bases, boots = [], []
    for df, mot, vib, fq_ in zip(dfs, mots, vibs, freqs):
        b, s = bootstrap(df, mot, vib, fq_, rng=rng)
        bases.append(b); boots.append(s)

    uuids = {u.msg_info_dict.get("sys_uuid", "?") for u in ulogs}
    mesmo_veiculo = len(uuids) == 1 or args.split is not None

    L = ["# Comparacao de voos\n"]
    L.append("| voo | veiculo | firmware | tempo em modo automatico |")
    L.append("|---|---|---|---|")
    for nome, u, df in zip(nomes, ulogs, dfs):
        L.append(f"| {nome} | {u.msg_info_dict.get('ver_hw','?')} "
                 f"`{str(u.msg_info_dict.get('sys_uuid',''))[-8:]}` | "
                 f"{u.msg_info_dict.get('ver_sw_branch','?')} | {len(df)/HZ:.0f} s |")
    L.append("")
    if not mesmo_veiculo:
        L.append("> **Atencao: os logs nao sao do mesmo veiculo** (`sys_uuid` difere). "
                 "A comparacao abaixo mistura diferencas de configuracao com "
                 "diferencas de aeronave.\n")

    # ------------------------------------------------- diff de parametros
    if args.split is None:
        tuning, outros = diff_parametros(ulogs, nomes)
        L.append("## O que mudou na configuracao\n")
        if tuning:
            L.append("Parametros de controle/estimacao/atuacao:\n")
            L.append("| parametro | " + " | ".join(nomes) + " |")
            L.append("|---" * (len(nomes) + 1) + "|")
            for k, vals in tuning:
                L.append(f"| `{k}` | " + " | ".join(
                    f"{v:.4g}" if isinstance(v, float) else str(v) for v in vals) + " |")
        else:
            L.append("- nenhum parametro de controle/estimacao mudou entre os voos")
        L.append(f"\n- outros parametros diferentes (telemetria, RC, etc.): "
                 f"{len(outros)}")
        if outros[:10]:
            L.append("  - " + ", ".join(f"`{k}`" for k, _ in outros[:10]))
        L.append("")

    # ------------------------------------------------------- comparacao
    L.append("## O que mudou no voo\n")
    L.append("Medido so em modos automaticos. IC = intervalo de 90% por bootstrap "
             "de blocos de 10 s.\n")
    cab = "| metrica | " + " | ".join(nomes) + " |"
    if len(nomes) == 2:
        cab = "| metrica | " + " | ".join(nomes) + " | diferenca (IC 90%) | |"
    L.append(cab)
    L.append("|---" * (len(cab.split("|")) - 2) + "|")

    destaques = []
    for k, (rot, uni, menor_melhor) in METRICAS.items():
        if all(not np.isfinite(b.get(k, np.nan)) for b in bases):
            continue
        sem_ic = k in SEM_IC
        cels = []
        for b, s in zip(bases, boots):
            v = b.get(k, np.nan)
            if sem_ic:
                cels.append(f"{v:.3g}")
                continue
            lo, hi = (np.nanpercentile(s[k], 5), np.nanpercentile(s[k], 95)) \
                if k in s and np.isfinite(s[k]).any() else (np.nan, np.nan)
            cels.append(f"{v:.3g} [{lo:.3g}–{hi:.3g}]")
        linha = f"| {rot} [{uni}] | " + " | ".join(cels) + " |"
        if len(nomes) == 2 and not sem_ic and k in boots[0] and k in boots[1]:
            d = boots[1][k] - boots[0][k]
            d = d[np.isfinite(d)]
            if d.size:
                lo, hi = np.nanpercentile(d, 5), np.nanpercentile(d, 95)
                delta = bases[1].get(k, np.nan) - bases[0].get(k, np.nan)
                sig = ((lo > 0) or (hi < 0)) and np.isfinite(delta)
                if sig and menor_melhor is not None:
                    marca = "melhor" if (delta < 0) == menor_melhor else "pior"
                    destaques.append((rot, delta, uni, marca, k))
                elif sig:
                    marca = "mudou"
                    destaques.append((rot, delta, uni, marca, k))
                else:
                    marca = "—"
                if not np.isfinite(delta):
                    linha += " nao estimavel nos dois voos | — |"
                else:
                    linha += f" {delta:+.3g} [{lo:+.3g}–{hi:+.3g}] | {marca} |"
            else:
                linha += " – | – |"
        elif len(nomes) == 2:
            d = bases[1].get(k, np.nan) - bases[0].get(k, np.nan)
            linha += f" {d:+.3g} (sem IC) | — |"
        L.append(linha)
    L.append("")

    if len(nomes) == 2:
        mm = [b.get("motor_medio", np.nan) for b in bases]
        if np.isfinite(mm).all() and abs(mm[1] - mm[0]) / max(mm[0], 1e-6) > 0.05:
            L.append(f"> **Comparabilidade:** o comando medio de motor mudou "
                     f"{100*(mm[1]-mm[0])/mm[0]:+.0f}% ({mm[0]:.3f} -> {mm[1]:.3f}). "
                     f"Isso e peso, bateria ou eficiencia de propulsao diferente — "
                     f"os dois voos nao estao na mesma condicao, e nenhuma "
                     f"comparacao de desempenho abaixo deve ser atribuida so a "
                     f"configuracao.\n")
        L.append("### Leitura\n")
        exc_mudou = any(k == "exc_p" for *_, k in destaques)
        robustas = [d for d in destaques if d[4] in ROBUSTAS]
        sensiveis = [d for d in destaques if d[4] in SENSIVEIS]
        outras = [d for d in destaques if d[4] not in ROBUSTAS | SENSIVEIS]
        if not destaques:
            L.append("- nenhuma metrica mudou de forma significativa: as diferencas "
                     "cabem dentro da variacao interna de cada voo")
        if robustas:
            L.append("**Evidencia robusta** (metricas que nao escalam com o quanto "
                     "o voo exigiu da aeronave):\n")
            for rot, delta, uni, marca, _ in robustas:
                L.append(f"- **{rot}**: {delta:+.3g} {uni} — {marca}")
            L.append("")
        if sensiveis:
            L.append("**Depende das condicoes** (cresce junto com a agressividade "
                     "do voo):\n")
            for rot, delta, uni, marca, _ in sensiveis:
                L.append(f"- {rot}: {delta:+.3g} {uni} — {marca}")
            L.append("")
        if outras:
            L.append("**Contexto**:\n")
            for rot, delta, uni, marca, _ in outras:
                L.append(f"- {rot}: {delta:+.3g} {uni} — {marca}")
            L.append("")
        if exc_mudou and sensiveis:
            L.append("> A agressividade do voo mudou de forma significativa. "
                     "As metricas da lista 'depende das condicoes' estao "
                     "contaminadas por isso: um voo mais exigido produz mais erro "
                     "mesmo sem nenhuma mudanca de configuracao. Use a lista "
                     "robusta como evidencia.")
        else:
            L.append("> Significativo nao e o mesmo que causado pela mudanca de "
                     "configuracao: vento, carga e pilotagem tambem mudam entre "
                     "voos.")
        L.append("")

    # ------------------------------------------------------------ figura
    chaves = [k for k in ("err_roll", "err_pitch", "err_p", "err_q",
                          "err_p_norm", "err_q_norm", "atraso_p", "atraso_q",
                          "K_p", "K_q", "banda_p",
                          "vib_hf", "sat", "motor_medio", "exc_p")
              if any(np.isfinite(b.get(k, np.nan)) for b in bases)]
    ncol = 4
    nrow = int(np.ceil(len(chaves) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow),
                            layout="constrained")
    fig.patch.set_facecolor(BG)
    axs = np.atleast_1d(axs).ravel()
    for ax, k in zip(axs, chaves):
        rot, uni, _ = METRICAS[k]
        vals = [b.get(k, np.nan) for b in bases]
        los = [np.nanpercentile(s[k], 5) if np.isfinite(s[k]).any() else np.nan
               for s in boots]
        his = [np.nanpercentile(s[k], 95) if np.isfinite(s[k]).any() else np.nan
               for s in boots]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=[C[i % len(C)] for i in range(len(vals))], width=0.6)
        if k not in SEM_IC:
            err = np.vstack([np.array(vals) - np.array(los),
                             np.array(his) - np.array(vals)])
            ax.errorbar(x, vals, yerr=np.abs(err), fmt="none", ecolor=INK,
                        capsize=4, lw=1)
        ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=7, rotation=12)
        ax.set_title(f"{rot} [{uni}]", fontsize=8.5, color=INK, loc="left")
        ax.set_facecolor(BG); ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=7)
        for s in ax.spines.values():
            s.set_color(GRID)
    for ax in axs[len(chaves):]:
        ax.axis("off")
    fig.suptitle("Comparacao de voos — barras com intervalo de 90% (bootstrap)",
                 fontsize=11, color=INK, x=0.01, ha="left")

    os.makedirs(args.out, exist_ok=True)
    tag = "_".join(n.replace("[", "").replace("]", "").replace(":", "")[:8]
                   for n in nomes)
    png = os.path.join(args.out, f"comparacao_{tag}.png")
    md = os.path.join(args.out, f"comparacao_{tag}.md")
    fig.savefig(png, dpi=130)
    L.append(f"![comparacao]({os.path.basename(png)})")
    open(md, "w").write("\n".join(L) + "\n")
    print(f"relatorio: {md}")
    print(f"figura:    {png}")
    for rot, delta, uni, marca, k in destaques:
        tag = ("robusta" if k in ROBUSTAS else
               "condicao" if k in SENSIVEIS else "contexto")
        print(f"  [{tag:8}] {marca:7} {rot}: {delta:+.3g} {uni}")
    if len(nomes) == 2 and not destaques:
        print("  nenhuma diferenca significativa")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Identificacao no dominio da frequencia, com coerencia.

Por que: derivar no tempo amplifica ruido; na frequencia derivar e multiplicar
por jw, entao a resposta em frequencia sai direto dos espectros cruzados. E,
principalmente, a **coerencia** diz em que faixa o dado e confiavel — algo que
nenhuma metrica do dominio do tempo entrega.

Para cada eixo estima-se  H(f) = Pxy / Pxx  entre o torque comandado e a taxa
angular, com coerencia  gamma^2 = |Pxy|^2 / (Pxx * Pyy).

Depois ajusta-se, so onde a coerencia passa do limiar, o modelo de 1a ordem

    H(jw) = K / (jw + a)

cuja forma no tempo e  w_dot = K*tau - a*w:  K e a efetividade de controle e
`a` e o amortecimento. O ajuste e linear em (a, K):

    jw * H(jw)  =  K - a * H(jw)

Isso da uma estimativa do amortecimento **independente** da regressao no tempo.

Uso: python src/freq_id.py data/ulg/<id>/<id>.ulg [--modes Manual] [--coer 0.6]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import coherence, csd, welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import sindy_rates as sr  # noqa: E402

C_A, C_B = "#1F6FB2", "#D1670B"
INK, MUTED, GRID, BG = "#1a1a1a", "#6b6b6b", "#e3e3e0", "#fcfcfb"


def ajusta_primeira_ordem(f, H, w8):
    """Minimos quadrados complexos de jw*H = K - a*H, ponderado pela coerencia."""
    w = 2 * np.pi * f
    y = 1j * w * H
    A = np.column_stack([-H, np.ones_like(H)])          # [a, K]
    W = np.sqrt(w8)
    Ar = np.vstack([np.real(A) * W[:, None], np.imag(A) * W[:, None]])
    yr = np.concatenate([np.real(y) * W, np.imag(y) * W])
    sol, *_ = np.linalg.lstsq(Ar, yr, rcond=None)
    a, K = float(sol[0]), float(sol[1])
    Hfit = K / (1j * w + a)
    num = np.sum(w8 * np.abs(H - Hfit) ** 2)
    den = np.sum(w8 * np.abs(H - np.mean(H)) ** 2)
    return a, K, float(1 - num / den) if den > 0 else np.nan, Hfit


def compara_estruturas(fb, Hb, wb):
    """Compara estruturas candidatas para a planta torque -> taxa.

    O multirrotor nu e, em primeira aproximacao, um INTEGRADOR (torque -> taxa)
    com atraso de atuador; amortecimento aerodinamico so aparece se for grande o
    bastante. Ajustar 1a ordem com polo livre forca um polo onde talvez nao haja
    — e foi assim que apareceu amortecimento negativo.
    """
    s_ = 1j * 2 * np.pi * fb

    def ganho_otimo(base):
        K = np.real(np.sum(wb * np.conj(base) * Hb) / np.sum(wb * np.abs(base) ** 2))
        num = np.sum(wb * np.abs(Hb - K * base) ** 2)
        den = np.sum(wb * np.abs(Hb - np.mean(Hb)) ** 2)
        return K, float(1 - num / den)

    saida = []
    melhor = None
    for td in np.arange(0.0, 0.201, 0.005):
        a_, K_, r_, _ = ajusta_primeira_ordem(fb, Hb * np.exp(1j * 2 * np.pi * fb * td), wb)
        if melhor is None or r_ > melhor[0]:
            melhor = (r_, a_, K_, td)
    saida.append(("K e^(-s td)/(s+a), polo livre",
                  f"a={melhor[1]:+.2f} 1/s, td={1000*melhor[3]:.0f} ms, K={melhor[2]:.0f}",
                  melhor[0]))
    melhor = None
    for Tm in np.arange(0.005, 0.201, 0.005):
        K, r = ganho_otimo(1.0 / (s_ * (1 + s_ * Tm)))
        if melhor is None or r > melhor[0]:
            melhor = (r, Tm, K)
    saida.append(("K/(s(1+s Tm)) integrador+motor",
                  f"Tm={1000*melhor[1]:.0f} ms, K={melhor[2]:.0f}", melhor[0]))
    melhor = None
    for td in np.arange(0.0, 0.201, 0.005):
        K, r = ganho_otimo(np.exp(-s_ * td) / s_)
        if melhor is None or r > melhor[0]:
            melhor = (r, td, K)
    saida.append(("K e^(-s td)/s integrador puro",
                  f"td={1000*melhor[1]:.0f} ms, K={melhor[2]:.0f}", melhor[0]))
    melhor = None
    for Tm in np.arange(0.005, 0.151, 0.005):
        for td in np.arange(0.0, 0.101, 0.005):
            K, r = ganho_otimo(np.exp(-s_ * td) / (s_ * (1 + s_ * Tm)))
            if melhor is None or r > melhor[0]:
                melhor = (r, Tm, td, K)
    saida.append(("K e^(-s td)/(s(1+s Tm))",
                  f"Tm={1000*melhor[1]:.0f}, td={1000*melhor[2]:.0f} ms, "
                  f"K={melhor[3]:.0f}", melhor[0]))
    return saida


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=20.0)
    ap.add_argument("--modes", nargs="*", default=["Manual"])
    ap.add_argument("--lag", type=float, default=0.0,
                    help="atraso aplicado a entrada antes do calculo [s]")
    ap.add_argument("--nperseg", type=float, default=8.0, help="segmento de Welch [s]")
    ap.add_argument("--coer", type=float, default=0.6, help="limiar de coerencia")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status"])
    t, X, Xdot, U, g = sr.build_dataset(ulog, args.hz, args.fc, args.modes,
                                        None, None, "torque")
    if args.lag:
        U = sr.first_order_lag(U, args.lag, args.hz)
    nome = os.path.basename(args.ulg)[:8]
    nper = int(args.nperseg * args.hz)

    print("=" * 78)
    print(f"Identificacao em frequencia — {nome}")
    print(f"{len(t)} amostras @ {args.hz:g} Hz, modos={args.modes}, "
          f"segmento de Welch {args.nperseg:g} s "
          f"(resolucao {1/args.nperseg:.3f} Hz), limiar de coerencia {args.coer}")

    eixos = [("p", 0), ("q", 1), ("r", 2)]
    fig, axs = plt.subplots(3, 3, figsize=(14, 9), layout="constrained")
    fig.patch.set_facecolor(BG)
    resumo = []
    for row, (nm, i) in enumerate(eixos):
        x = U[:, i]                      # torque comandado
        y = X[:, i]                      # taxa medida
        f, Pxx = welch(x, fs=args.hz, nperseg=nper)
        _, Pyy = welch(y, fs=args.hz, nperseg=nper)
        _, Pxy = csd(x, y, fs=args.hz, nperseg=nper)
        _, coh = coherence(x, y, fs=args.hz, nperseg=nper)
        H = Pxy / np.where(Pxx == 0, np.nan, Pxx)

        banda = (f > 0.05) & (f < args.fc) & (coh >= args.coer)
        if banda.sum() >= 5:
            fb, Hb, wb = f[banda], H[banda], coh[banda]
            # varredura do atraso puro: H = K*exp(-jw*td)/(jw + a).
            # Compensar o atraso antes do ajuste de 1a ordem torna o problema linear.
            melhor = None
            for td in np.arange(0.0, 0.201, 0.005):
                Hc = Hb * np.exp(1j * 2 * np.pi * fb * td)
                a_, K_, r_, _ = ajusta_primeira_ordem(fb, Hc, wb)
                if melhor is None or r_ > melhor[2]:
                    melhor = (a_, K_, r_, td)
            a, K, r2f, td = melhor
            fmin, fmax = fb.min(), fb.max()
            ok = True
        else:
            a = K = r2f = td = np.nan
            fmin = fmax = np.nan
            ok = False
        resumo.append((nm, a, K, r2f, fmin, fmax, int(banda.sum()),
                       float(coh.max()), td))

        ax = axs[row, 0]
        ax.loglog(f[1:], np.abs(H[1:]), color=C_A, lw=1.2)
        if ok:
            ax.loglog(f[banda], np.abs(K / (1j * 2 * np.pi * f[banda] + a)),
                      color=C_B, lw=1.4, ls="--")
        ax.set_ylabel(f"|H| {nm}", fontsize=9)
        ax.set_title(f"ganho {nm}: torque -> taxa", fontsize=9, loc="left")

        ax = axs[row, 1]
        ax.semilogx(f[1:], np.angle(H[1:], deg=True), color=C_A, lw=1.2)
        if ok:
            ax.semilogx(f[banda],
                        np.angle(K * np.exp(-1j * 2 * np.pi * f[banda] * td)
                                 / (1j * 2 * np.pi * f[banda] + a), deg=True),
                        color=C_B, lw=1.4, ls="--")
        ax.set_ylabel("fase [deg]", fontsize=9)
        ax.set_title(f"fase {nm}", fontsize=9, loc="left")

        ax = axs[row, 2]
        ax.semilogx(f[1:], coh[1:], color=C_A, lw=1.2)
        ax.axhline(args.coer, color=C_B, lw=1, ls="--")
        ax.set_ylim(0, 1); ax.set_ylabel("coerencia", fontsize=9)
        ax.set_title(f"coerencia {nm}", fontsize=9, loc="left")

        for ax in axs[row]:
            ax.set_facecolor(BG); ax.grid(True, which="both", color=GRID, lw=0.5)
            ax.tick_params(colors=MUTED, labelsize=8)
            for s in ax.spines.values():
                s.set_color(GRID)
            if row == 2:
                ax.set_xlabel("frequencia [Hz]", fontsize=9)

    print(f"\n{'eixo':5} {'banda coerente':>18} {'pts':>5} {'coer max':>9} "
          f"{'atraso ms':>10} {'a (amort.) 1/s':>15} {'K (ganho)':>11} {'R2':>7}")
    for nm, a, K, r2f, fmin, fmax, n, cmax, td in resumo:
        faixa = f"{fmin:.2f}-{fmax:.2f} Hz" if np.isfinite(fmin) else "nenhuma"
        print(f"{nm:5} {faixa:>18} {n:5} {cmax:9.2f} {1000*td:10.0f} "
              f"{a:15.3f} {K:11.1f} {r2f:7.3f}")
    print("\n`a` positivo = amortecimento fisico; negativo = nao fisico "
          "(residuo de malha fechada ou modelo inadequado).")

    print("\n--- qual estrutura descreve a planta? (R2 ponderado pela coerencia)")
    for nm, i in eixos[:2]:
        x, y = U[:, i], X[:, i]
        f, Pxx = welch(x, fs=args.hz, nperseg=nper)
        _, Pxy = csd(x, y, fs=args.hz, nperseg=nper)
        _, coh = coherence(x, y, fs=args.hz, nperseg=nper)
        H = Pxy / np.where(Pxx == 0, np.nan, Pxx)
        b = (f > 0.25) & (f < args.fc) & (coh >= args.coer)
        if b.sum() < 5:
            print(f"  {nm}: banda coerente insuficiente")
            continue
        print(f"  eixo {nm}:")
        for modelo, par, r in compara_estruturas(f[b], H[b], coh[b]):
            print(f"    {modelo:34} {par:34} R2={r:6.3f}")

    fig.suptitle(f"Resposta em frequencia e coerencia — {nome}", fontsize=11,
                 color=INK, x=0.01, ha="left")
    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"freq_{nome}.png")
    fig.savefig(png, dpi=130)
    print(f"figura: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

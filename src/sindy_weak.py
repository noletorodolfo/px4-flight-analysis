"""SINDy em forma fraca (integral): identificacao sem derivar o dado.

Em vez de estimar dx/dt numericamente (que amplifica ruido), multiplica-se a
equacao por uma funcao teste phi com suporte compacto e integra-se por partes:

    integral( phi * x_dot ) dt  =  -integral( phi' * x ) dt

O lado esquerdo, que exigia a derivada, vira uma integral do proprio sinal.
O lado direito,  integral( phi * Theta(x,u) ) dt, tambem e so integracao.
Nenhuma derivada numerica entra na regressao.

Este modulo refaz, em forma fraca, as duas analises feitas no dominio do tempo:
  - ajuste com biblioteca livre (compara com src/sindy_rates.py);
  - modelo fisico de Euler com razoes de inercia compartilhadas e varredura da
    superficie de custo (compara com src/sindy_physical.py).

Uso:
  python src/sindy_weak.py data/ulg/<id>/<id>.ulg [--hz 50] [--modes Manual]
                           [--janela 1.0] [--lag 0.06]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import sindy_rates as sr  # noqa: E402
from sindy_physical import inertia_coefs, r2  # noqa: E402

R2D = 180.0 / np.pi
C_A, C_B = "#1F6FB2", "#D1670B"
INK, MUTED, GRID, BG = "#1a1a1a", "#6b6b6b", "#e3e3e0", "#fcfcfb"


def funcao_teste(L: int, hz: float, p: int = 6):
    """phi(tau) = (1 - tau^2)^p em [-1, 1], com phi(+-1) = phi'(+-1) = 0."""
    tau = np.linspace(-1.0, 1.0, L)
    phi = (1.0 - tau ** 2) ** p
    dphi_dtau = -2.0 * p * tau * (1.0 - tau ** 2) ** (p - 1)
    meia_janela_s = (L - 1) / hz / 2.0
    dphi_dt = dphi_dtau / meia_janela_s
    return phi, dphi_dt


def montar_fraco(cols: np.ndarray, alvo: np.ndarray, hz: float,
                 janela_s: float = 1.0, passo_frac: float = 0.5, p: int = 6):
    """Monta A (integral de phi*coluna) e b (-integral de phi'*alvo).

    cols: (n, m) regressores;  alvo: (n, k) estados cuja derivada e modelada.
    Devolve A (W, m), b (W, k) e os indices centrais de cada janela.
    """
    n = cols.shape[0]
    L = max(8, int(round(janela_s * hz)))
    passo = max(1, int(round(L * passo_frac)))
    phi, dphi = funcao_teste(L, hz)
    norm = np.trapezoid(phi, dx=1.0 / hz)      # normaliza: unidades do sinal
    A, b, centros = [], [], []
    for s0 in range(0, n - L + 1, passo):
        sl = slice(s0, s0 + L)
        A.append(np.trapezoid(phi[:, None] * cols[sl], dx=1.0 / hz, axis=0) / norm)
        b.append(-np.trapezoid(dphi[:, None] * alvo[sl], dx=1.0 / hz, axis=0) / norm)
        centros.append(s0 + L // 2)
    return np.array(A), np.array(b), np.array(centros)


def biblioteca_cols(X, U):
    Theta, nomes = sr.library(X, U)
    return Theta, nomes


def cols_fisicas(X, U, i, a_i):
    """Colunas do modelo de Euler no eixo i, com o termo giroscopico ja fixado."""
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    T = U[:, 3]
    gyro = [q * r, p * r, p * q][i]
    tau = U[:, i]
    estado = [p, q, r][i]
    livres = np.column_stack([tau, tau * T, -estado])   # b0, b1, c
    fixo = a_i * gyro                                    # entra no lado conhecido
    return livres, fixo


def ajuste_fisico(X, U, hz, janela_s, ky, kz, mask_jan=None):
    """Ajusta b0,b1,c por eixo em forma fraca, dado (ky,kz). Devolve SSE e betas."""
    a = inertia_coefs(ky, kz)
    sse_total, betas, preds, alvos = 0.0, [], [], []
    for i in range(3):
        livres, fixo = cols_fisicas(X, U, i, a[i])
        cols = np.column_stack([livres, fixo])
        A, b, _ = montar_fraco(cols, X[:, [i]], hz, janela_s)
        if mask_jan is not None:
            A, b = A[mask_jan], b[mask_jan]
        y = b[:, 0] - A[:, 3]            # move o termo giroscopico para o alvo
        M = A[:, :3]
        res = lsq_linear(M, y, bounds=(np.zeros(3), np.full(3, np.inf)))
        betas.append(res.x)
        sse_total += float(np.sum(res.fun ** 2))
        preds.append(M @ res.x)
        alvos.append(y)
    return sse_total, betas, preds, alvos


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=None)
    ap.add_argument("--modes", nargs="*", default=["Manual"])
    ap.add_argument("--janela", type=float, default=1.0, help="largura da funcao teste [s]")
    ap.add_argument("--lag", type=float, default=0.06)
    ap.add_argument("--input", choices=["auto", "torque", "motors"], default="torque")
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--chunks", type=int, default=8)
    args = ap.parse_args(argv)

    fc = args.fc if args.fc else min(15.0, 0.2 * args.hz)
    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status"])
    t, X, Xdot, U, g = sr.build_dataset(ulog, args.hz, fc, args.modes, None, None,
                                        args.input)
    U = sr.first_order_lag(U, args.lag, args.hz)
    nome = os.path.basename(args.ulg)[:8]

    print("=" * 78)
    print(f"SINDy em FORMA FRACA — {nome}")
    print(f"entrada={g['source']}  atraso={args.lag*1000:.0f} ms  grade={args.hz:g} Hz  "
          f"passa-baixa={fc:g} Hz  janela de teste={args.janela:g} s  "
          f"{len(t)} amostras")

    # ---------------------------------------------- biblioteca livre, fraca
    Theta, nomes = biblioteca_cols(X, U)
    A, B, centros = montar_fraco(Theta, X, args.hz, args.janela)
    tj = t[centros]
    tr_j, te_j = sr.block_split(tj, 10.0)
    print(f"janelas de teste: {len(A)} ({tr_j.sum()} treino / {te_j.sum()} teste)")

    print("\n--- biblioteca livre: forma fraca vs forma forte (derivada)")
    for k, eixo in enumerate(["p_dot", "q_dot", "r_dot"]):
        # fraca
        melhor = None
        for th in [0.002, 0.01, 0.05, 0.1, 0.2, 0.35]:
            xi, _ = sr.stlsq(A[tr_j], B[tr_j, k], th)
            sc = r2(B[te_j, k], A[te_j] @ xi)
            n_t = int(np.sum(np.abs(xi) > 0))
            if melhor is None or sc > melhor[0]:
                melhor = (sc, th, xi, n_t)
        sc_w, th_w, xi_w, nt_w = melhor
        # forte, no mesmo recorte
        tr_s, te_s = sr.block_split(t, 10.0)
        melhor_s = None
        for th in [0.002, 0.01, 0.05, 0.1, 0.2, 0.35]:
            xi, _ = sr.stlsq(Theta[tr_s], Xdot[tr_s, k], th)
            sc = r2(Xdot[te_s, k], Theta[te_s] @ xi)
            if melhor_s is None or sc > melhor_s[0]:
                melhor_s = (sc, th, xi, int(np.sum(np.abs(xi) > 0)))
        sc_s, _, xi_s, nt_s = melhor_s
        # a previsao do modelo fraco avaliada contra a derivada (comparavel)
        sc_cruz = r2(Xdot[te_s, k], Theta[te_s] @ xi_w)
        print(f"  {eixo}: R2 fraca {sc_w:6.3f} ({nt_w} termos) | "
              f"R2 forte {sc_s:6.3f} ({nt_s} termos) | "
              f"modelo fraco avaliado na derivada: {sc_cruz:6.3f}")

    # ------------------------------------- modelo fisico em forma fraca
    print("\n--- modelo fisico de Euler, agora em forma fraca")

    def obj(z):
        return ajuste_fisico(X, U, args.hz, args.janela, np.exp(z[0]), np.exp(z[1]))[0]

    best = None
    for ky0, kz0 in [(1.0, 2.0), (1.0, 1.0), (1.2, 2.2), (0.8, 1.5), (1.0, 0.5)]:
        res = minimize(obj, np.log([ky0, kz0]), method="L-BFGS-B",
                       bounds=[(np.log(0.3), np.log(3.0)), (np.log(0.3), np.log(4.0))])
        if best is None or res.fun < best.fun:
            best = res
    ky, kz = np.exp(best.x)
    a = inertia_coefs(ky, kz)
    sse_livre, betas, preds, alvos = ajuste_fisico(X, U, args.hz, args.janela, ky, kz)
    sse_plano = ajuste_fisico(X, U, args.hz, args.janela, 1.0, 2.0)[0]
    print(f"  razoes ajustadas: Iy/Ix = {ky:.3f}   Iz/Ix = {kz:.3f}")
    print(f"  coeficientes: (Iy-Iz)/Ix={a[0]:+.3f}  (Iz-Ix)/Iy={a[1]:+.3f}  "
          f"(Ix-Iy)/Iz={a[2]:+.3f}")
    print(f"  SSE no otimo livre = {sse_livre:.6g};  em (1, 2) = {sse_plano:.6g} "
          f"({100*(sse_plano-sse_livre)/sse_livre:+.2f}%)")
    for i, nm in enumerate(["p_dot", "q_dot", "r_dot"]):
        b0, b1, c = betas[i]
        print(f"    {nm}: b0={b0:8.2f}  b1={b1:8.2f}  c={c:7.3f} 1/s   "
              f"R2(fraco)={r2(alvos[i], preds[i]):6.3f}")

    # teste decisivo em forma fraca
    print("\n--- o termo giroscopico fisico (Iy/Ix=1, Iz/Ix=2) ajuda na forma fraca?")
    a_ref = inertia_coefs(1.0, 2.0)
    for i, nm in enumerate(["p_dot", "q_dot", "r_dot"]):
        _, _, pr_c, al_c = ajuste_fisico(X, U, args.hz, args.janela, 1.0, 2.0)
        _, _, pr_s, al_s = ajuste_fisico(X, U, args.hz, args.janela, 1.0, 1.0)  # a=0 em p_dot
        rc = r2(al_c[i], pr_c[i])
        # modelo sem termo giroscopico: a = 0 em todos os eixos
        livres, _ = cols_fisicas(X, U, i, 0.0)
        A0, b0_, _ = montar_fraco(np.column_stack([livres, np.zeros(len(X))]),
                                  X[:, [i]], args.hz, args.janela)
        res0 = lsq_linear(A0[:, :3], b0_[:, 0], bounds=(np.zeros(3), np.full(3, np.inf)))
        rn = r2(b0_[:, 0], A0[:, :3] @ res0.x)
        ver = ("ajuda" if rc - rn > 0.005 else
               "indiferente" if rc - rn > -0.005 else "ATRAPALHA")
        print(f"  {nm}: com termo R2 = {rc:.4f}   sem termo R2 = {rn:.4f}   "
              f"ganho = {rc-rn:+.4f}  -> {ver}")

    # estabilidade por trecho
    print("\n--- estabilidade por trecho (forma fraca)")
    nj = len(A)
    blocos = np.array_split(np.arange(nj), args.chunks)
    kys, kzs = [], []
    for ci, ids in enumerate(blocos):
        m = np.zeros(nj, dtype=bool); m[ids] = True

        def objc(z, m=m):
            return ajuste_fisico(X, U, args.hz, args.janela,
                                 np.exp(z[0]), np.exp(z[1]), mask_jan=m)[0]

        rc = minimize(objc, np.log([1.0, 1.0]), method="L-BFGS-B",
                      bounds=[(np.log(0.3), np.log(3.0)), (np.log(0.3), np.log(4.0))])
        k1, k2 = np.exp(rc.x)
        kys.append(k1); kzs.append(k2)
        print(f"  trecho {ci+1} ({tj[ids[0]]:6.1f}-{tj[ids[-1]]:6.1f} s): "
              f"Iy/Ix={k1:5.2f}  Iz/Ix={k2:5.2f}")
    kys, kzs = np.array(kys), np.array(kzs)
    print(f"  dispersao: Iy/Ix = {kys.mean():.2f} +- {kys.std():.2f}   "
          f"Iz/Ix = {kzs.mean():.2f} +- {kzs.std():.2f}")

    # superficie de custo
    n = args.grid
    gky = np.linspace(0.4, 2.2, n); gkz = np.linspace(0.4, 3.6, n)
    Z = np.empty((n, n))
    for i, kzi in enumerate(gkz):
        for j, kyj in enumerate(gky):
            Z[i, j] = ajuste_fisico(X, U, args.hz, args.janela, kyj, kzi)[0]
    rel = (Z - Z.min()) / Z.min() * 100
    print(f"\n--- superficie de custo (fraca): piora maxima {rel.max():.2f}%, "
          f"{100*np.mean(rel < 1.0):.1f}% da faixa a menos de 1% do minimo")

    fig, ax = plt.subplots(figsize=(7.2, 5.4), layout="constrained")
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    cs = ax.contourf(gky, gkz, np.clip(rel, 0, 5), levels=20, cmap="BuPu_r")
    ax.contour(gky, gkz, rel, levels=[0.1, 0.5, 1.0], colors=[C_B], linewidths=1.0)
    ax.plot(kys, kzs, "x", color="#444", ms=6, mew=1.4, label="ajuste por trecho")
    ax.plot([ky], [kz], "o", color=C_B, ms=7, label="ajuste (forma fraca)")
    ax.plot([1.0], [2.0], "s", color=C_A, ms=7, label="quad plano (Iz=Ix+Iy)")
    ax.set_xlabel("Iy / Ix", fontsize=9); ax.set_ylabel("Iz / Ix", fontsize=9)
    ax.set_title("forma fraca — piora do SSE [%] no plano das inercias",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.colorbar(cs, ax=ax, label="% acima do minimo")
    out = os.path.join("out", f"inercia_fraca_{nome}.png")
    fig.savefig(out, dpi=130)
    print(f"figura: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

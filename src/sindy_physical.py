"""Identificacao com biblioteca restrita a termos fisicamente admissiveis.

Modelo (Euler, corpo rigido, eixos principais), com as tres equacoes acopladas
por um unico par de razoes de inercia ky = Iy/Ix, kz = Iz/Ix:

    p_dot = (ky - kz)     * q*r + (bx0 + bx1*T) * tau_x - cx * p
    q_dot = (kz - 1)/ky   * p*r + (by0 + by1*T) * tau_y - cy * q
    r_dot = (1 - ky)/kz   * p*q + (bz0 + bz1*T) * tau_z - cz * r

Restricoes impostas: efetividade de controle positiva (b >= 0), amortecimento
aerodinamico nao negativo (c >= 0) e razoes de inercia positivas. Nao ha termo
constante nem termos cruzados sem justificativa fisica.

Dado (ky, kz), o problema e linear nos demais parametros -> o ajuste e uma
busca em duas dimensoes com minimos quadrados limitados por dentro. Isso
permite varrer a superficie de custo e dizer se as inercias sao de fato
identificaveis neste voo, em vez de so reportar um numero.

Uso:
  python src/sindy_physical.py data/ulg/<id>/<id>.ulg [--hz 50] [--modes Manual]
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

R2D = 180.0 / np.pi
C_MEAS, C_MODEL = "#1F6FB2", "#D1670B"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e0"


def inertia_coefs(ky: float, kz: float) -> np.ndarray:
    """Coeficientes giroscopicos a partir das razoes de inercia (Ix = 1)."""
    return np.array([ky - kz, (kz - 1.0) / ky, (1.0 - ky) / kz])


def fit_linear_part(X, Xdot, U, a, idx, bounded=True):
    """Com os coeficientes giroscopicos fixos, ajusta [b0, b1, c] de um eixo."""
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    T = U[:, 3]
    gyro = [q * r, p * r, p * q][idx]
    tau = U[:, idx]
    state = [p, q, r][idx]
    A = np.column_stack([tau, tau * T, -state])
    y = Xdot[:, idx] - a[idx] * gyro
    if bounded:
        res = lsq_linear(A, y, bounds=(np.zeros(3), np.full(3, np.inf)))
        beta, resid = res.x, res.fun
    else:
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = A @ beta - y
    return beta, resid, A, y


def cost(ky, kz, X, Xdot, U, mask, bounded=True):
    a = inertia_coefs(ky, kz)
    tot, parts = 0.0, []
    for i in range(3):
        beta, resid, _, _ = fit_linear_part(X[mask], Xdot[mask], U[mask], a, i, bounded)
        sse = float(np.sum(resid ** 2))
        tot += sse
        parts.append((beta, sse))
    return tot, parts


def r2(y, yhat):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else np.nan


def predict(X, U, a, betas):
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    T = U[:, 3]
    gyros = [q * r, p * r, p * q]
    states = [p, q, r]
    out = []
    for i in range(3):
        b0, b1, c = betas[i]
        out.append(a[i] * gyros[i] + (b0 + b1 * T) * U[:, i] - c * states[i])
    return np.column_stack(out)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=None)
    ap.add_argument("--modes", nargs="*", default=["Manual"])
    ap.add_argument("--lag", type=float, default=0.06)
    ap.add_argument("--input", choices=["auto", "torque", "motors"], default="torque")
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--chunks", type=int, default=6)
    args = ap.parse_args(argv)

    fc = args.fc if args.fc else min(15.0, 0.2 * args.hz)
    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status"])
    t, X, Xdot, U, g = sr.build_dataset(ulog, args.hz, fc, args.modes, None, None,
                                        args.input)
    U = sr.first_order_lag(U, args.lag, args.hz)
    tr, te = sr.block_split(t)
    name = os.path.basename(args.ulg)[:8]

    print("=" * 76)
    print(f"Modelo fisico restrito — {name}")
    print(f"entrada={g['source']}  atraso={args.lag*1000:.0f} ms  grade={args.hz:g} Hz  "
          f"{len(t)} amostras  modos={args.modes}")

    # ---------------------------------------------------------- ajuste
    def obj(z):
        return cost(np.exp(z[0]), np.exp(z[1]), X, Xdot, U, tr)[0]

    best = None
    for ky0, kz0 in [(1.0, 2.0), (1.0, 1.0), (1.2, 2.2), (0.8, 1.5), (1.0, 0.5)]:
        res = minimize(obj, np.log([ky0, kz0]), method="L-BFGS-B",
                       bounds=[(np.log(0.3), np.log(3.0)),
                               (np.log(0.3), np.log(4.0))])
        if best is None or res.fun < best.fun:
            best = res
    ky, kz = np.exp(best.x)
    sse_free = best.fun
    # ponto de referencia fisico: quad aproximadamente plano e simetrico
    sse_planar = cost(1.0, 2.0, X, Xdot, U, tr)[0]
    # desigualdade triangular do tensor de inercia: |Ix-Iy| <= Iz <= Ix+Iy
    tri_ok = abs(1.0 - ky) <= kz <= 1.0 + ky
    a = inertia_coefs(ky, kz)
    _, parts = cost(ky, kz, X, Xdot, U, tr)
    betas = [pp[0] for pp in parts]

    print(f"\nrazoes de inercia ajustadas (livre):  Iy/Ix = {ky:.3f}   Iz/Ix = {kz:.3f}")
    print(f"  desigualdade triangular (|Ix-Iy| <= Iz <= Ix+Iy): "
          f"{'satisfeita' if tri_ok else 'VIOLADA'}")
    print(f"  SSE no otimo livre          = {sse_free:.6g}")
    print(f"  SSE em (Iy/Ix=1, Iz/Ix=2)   = {sse_planar:.6g}  "
          f"({100*(sse_planar-sse_free)/sse_free:+.2f}% em relacao ao otimo)")
    print("  -> quad plano e simetrico tem Iz ~ Ix+Iy, isto e Iz/Ix ~ 2;"
          " Iz/Ix << 1 so existe em corpo alongado no eixo vertical")
    print(f"coeficientes giroscopicos:   (Iy-Iz)/Ix = {a[0]:+.3f}   "
          f"(Iz-Ix)/Iy = {a[1]:+.3f}   (Ix-Iy)/Iz = {a[2]:+.3f}")
    print("\nparametros por eixo (b0 + b1*T = efetividade; c = amortecimento):")
    axes_n = ["p_dot", "q_dot", "r_dot"]
    for i, nm in enumerate(axes_n):
        b0, b1, c = betas[i]
        print(f"  {nm}:  b0 = {b0:8.2f}   b1 = {b1:8.2f}   c = {c:7.3f} 1/s"
              + (f"   (tau = {1/c:.2f} s)" if c > 1e-6 else "   (sem amortecimento)"))

    Yhat = predict(X, U, a, betas)
    print("\najuste (R2):")
    for i, nm in enumerate(axes_n):
        print(f"  {nm}: treino {r2(Xdot[tr, i], Yhat[tr, i]):6.3f}   "
              f"teste {r2(Xdot[te, i], Yhat[te, i]):6.3f}   "
              f"residuo RMS {np.std(Xdot[te, i]-Yhat[te, i])*R2D:7.1f} deg/s^2")

    # ----------------------------------- identificabilidade das inercias
    print("\n--- o termo giroscopico e visivel no meio do ruido?")
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    gyros = [q * r, p * r, p * q]
    gyro_names = ["q*r", "p*r", "p*q"]
    for i, nm in enumerate(axes_n):
        contrib = np.std(a[i] * gyros[i]) * R2D
        resid = np.std(Xdot[te, i] - Yhat[te, i]) * R2D
        print(f"  {nm}: contribuicao de {gyro_names[i]:4} = {contrib:7.2f} deg/s^2   "
              f"residuo = {resid:7.1f} deg/s^2   razao = {contrib/resid:6.3f}")

    # superficie de custo (perfil) em (ky, kz)
    n = args.grid
    kys = np.linspace(0.4, 2.2, n)
    kzs = np.linspace(0.4, 3.6, n)
    Z = np.empty((n, n))
    for i, kzi in enumerate(kzs):
        for j, kyj in enumerate(kys):
            Z[i, j] = cost(kyj, kzi, X, Xdot, U, tr)[0]
    Zmin = Z.min()
    rel = (Z - Zmin) / Zmin * 100.0  # piora percentual do SSE
    inside = rel < 1.0
    print(f"\n--- superficie de custo em (Iy/Ix, Iz/Ix), grade {n}x{n}")
    print(f"  piora maxima do SSE em toda a faixa varrida: {rel.max():.2f}%")
    print(f"  fracao da faixa que fica a menos de 1% do minimo: "
          f"{100*inside.mean():.1f}%")
    if inside.any():
        print(f"  faixa de Iy/Ix dentro de 1%: "
              f"{kys[inside.any(axis=0)].min():.2f} a {kys[inside.any(axis=0)].max():.2f}")
        print(f"  faixa de Iz/Ix dentro de 1%: "
              f"{kzs[inside.any(axis=1)].min():.2f} a {kzs[inside.any(axis=1)].max():.2f}")

    # quanto de excitacao faltou
    print("\n--- excitacao necessaria para enxergar as inercias")
    for i, nm in enumerate(axes_n):
        resid = np.std(Xdot[te, i] - Yhat[te, i])
        need = 3 * resid / max(abs(a[i]), 1e-6)   # |gyro| para contribuir 3x o residuo
        have = np.std(gyros[i])
        print(f"  {nm}: std({gyro_names[i]}) atual = {have:7.4f} rad^2/s^2   "
              f"necessario ~{need:7.4f}   fator {need/max(have,1e-9):6.1f}x")

    # -------- teste decisivo: o termo giroscopico fisico melhora a previsao?
    print("\n--- o termo giroscopico com inercia de quad plano (Iy/Ix=1, Iz/Ix=2) ajuda?")
    a_ref = inertia_coefs(1.0, 2.0)
    for i, nm in enumerate(axes_n):
        b_with = fit_linear_part(X[tr], Xdot[tr], U[tr], a_ref, i)[0]
        b_none = fit_linear_part(X[tr], Xdot[tr], U[tr], np.zeros(3), i)[0]
        Bw = [b_with if j == i else np.zeros(3) for j in range(3)]
        Bn = [b_none if j == i else np.zeros(3) for j in range(3)]
        rw = r2(Xdot[te, i], predict(X, U, a_ref, Bw)[te, i])
        rn = r2(Xdot[te, i], predict(X, U, np.zeros(3), Bn)[te, i])
        verdict = "ajuda" if rw - rn > 0.005 else ("indiferente" if rw - rn > -0.005
                                                   else "ATRAPALHA")
        print(f"  {nm}: com termo R2 = {rw:.4f}   sem termo R2 = {rn:.4f}   "
              f"ganho = {rw-rn:+.4f}  -> {verdict}")

    # ------------------------- estabilidade: o mesmo numero sai de cada trecho?
    print("\n--- estabilidade da estimativa (ajuste independente por trecho)")
    nchunk = args.chunks
    edges = np.array_split(np.arange(len(t)), nchunk)
    kys_c, kzs_c = [], []
    for ci, ids in enumerate(edges):
        m = np.zeros(len(t), dtype=bool)
        m[ids] = True

        def objc(z, m=m):
            return cost(np.exp(z[0]), np.exp(z[1]), X, Xdot, U, m)[0]

        rc = minimize(objc, np.log([1.0, 1.0]), method="L-BFGS-B",
                      bounds=[(np.log(0.3), np.log(3.0)), (np.log(0.3), np.log(4.0))])
        kyc, kzc = np.exp(rc.x)
        kys_c.append(kyc); kzs_c.append(kzc)
        print(f"  trecho {ci+1} ({t[ids[0]]:6.1f}-{t[ids[-1]]:6.1f} s): "
              f"Iy/Ix = {kyc:5.2f}   Iz/Ix = {kzc:5.2f}")
    kys_c, kzs_c = np.array(kys_c), np.array(kzs_c)
    print(f"  dispersao: Iy/Ix = {kys_c.mean():.2f} +- {kys_c.std():.2f}   "
          f"Iz/Ix = {kzs_c.mean():.2f} +- {kzs_c.std():.2f}")
    print(f"  faixa: Iy/Ix {kys_c.min():.2f}..{kys_c.max():.2f}   "
          f"Iz/Ix {kzs_c.min():.2f}..{kzs_c.max():.2f}")

    # ------------------------------- comparacao com a biblioteca irrestrita
    print("\n--- comparacao com a biblioteca irrestrita (src/sindy_rates.py)")
    Theta, names = sr.library(X, U)
    for i, nm in enumerate(axes_n):
        xi, _ = sr.stlsq(Theta[tr], Xdot[tr, i], 0.1)
        r2u = r2(Xdot[te, i], Theta[te] @ xi)
        r2p = r2(Xdot[te, i], Yhat[te, i])
        print(f"  {nm}: fisico {r2p:6.3f}  vs  irrestrito {r2u:6.3f}   "
              f"(diferenca {r2u-r2p:+.3f}, {int(np.sum(np.abs(xi)>0))} termos)")

    # ------------------------------------------------------------ figura
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax = axs[0]
    cs = ax.contourf(kys, kzs, np.clip(rel, 0, 5), levels=20, cmap="BuPu_r")
    ax.contour(kys, kzs, rel, levels=[0.1, 0.5, 1.0], colors=[C_MODEL], linewidths=1.0)
    ax.plot([ky], [kz], "o", color=C_MODEL, ms=7, label="ajuste")
    ax.plot(kys_c, kzs_c, "x", color="#444", ms=6, mew=1.4,
            label=f"ajuste por trecho (n={nchunk})")
    ax.plot([1.0], [2.0], "s", color=C_MEAS, ms=7, label="quad plano ideal (Iz=Ix+Iy)")
    ax.set_xlabel("Iy / Ix", color=INK, fontsize=9)
    ax.set_ylabel("Iz / Ix", color=INK, fontsize=9)
    ax.set_title("piora do SSE [%] — plano das razoes de inercia",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.colorbar(cs, ax=ax, label="% acima do minimo")

    ax = axs[1]
    labels, contribs, resids = [], [], []
    for i, nm in enumerate(axes_n):
        labels.append(nm)
        contribs.append(np.std(a[i] * gyros[i]) * R2D)
        resids.append(np.std(Xdot[te, i] - Yhat[te, i]) * R2D)
    xpos = np.arange(3)
    ax.bar(xpos - 0.2, resids, 0.38, color=C_MEAS, label="residuo do modelo")
    ax.bar(xpos + 0.2, contribs, 0.38, color=C_MODEL, label="termo giroscopico")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, color=INK)
    ax.set_yscale("log")
    ax.set_ylabel("desvio padrao [deg/s²]", color=INK, fontsize=9)
    ax.set_title("o sinal que carrega a inercia esta abaixo do ruido",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    for a_ in axs:
        a_.set_facecolor("#fcfcfb")
        a_.tick_params(colors=MUTED, labelsize=8)
        for s in a_.spines.values():
            s.set_color(GRID)
    axs[1].grid(True, axis="y", color=GRID, lw=0.6)
    fig.suptitle(f"Inercias a partir do voo — {name}", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join("out", f"inercia_{name}.png")
    fig.savefig(out, dpi=130)
    print(f"\nfigura: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

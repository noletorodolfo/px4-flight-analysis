"""Identificacao com variavel instrumental (2SLS) para dado de malha fechada.

Problema: num voo normal o comando e gerado pelo controlador a partir do estado
(u = -K x). Regressores e residuo ficam correlacionados e os coeficientes saem
enviesados — foi assim que apareceu amortecimento POSITIVO (nao fisico) nos
ajustes por minimos quadrados comuns.

Remedio classico: usar um sinal **exogeno** como instrumento. Aqui o instrumento
e o comando do piloto (`manual_control_setpoint`), que entra no sistema de fora
e nao e calculado a partir do estado estimado. Fazemos minimos quadrados em dois
estagios (2SLS):

  estagio 1: projeta cada regressor no espaco dos instrumentos Z;
  estagio 2: regride a derivada nos regressores projetados.

O relatorio imprime a **forca do instrumento** (R2 do primeiro estagio): com
instrumento fraco a estimativa 2SLS e pior que a comum, e isso precisa ficar
visivel em vez de escondido.

Hipotese assumida (e discutivel): o piloto reage ao veiculo com atraso maior que
a escala de tempo da malha de taxas, entao o stick e aproximadamente exogeno.

Uso: python src/sindy_iv.py data/ulg/<id>/<id>.ulg [--modes Manual] [--lags 0 0.1 0.2 0.3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import sindy_rates as sr  # noqa: E402
from sindy_physical import inertia_coefs, r2  # noqa: E402

R2D = 180.0 / np.pi


def instrumentos(ulog, t_grid, hz, lags_s):
    """Matriz Z: canais do stick atrasados + constante."""
    mc = px4io.to_df(ulog, "manual_control_setpoint")
    canais = [c for c in ("roll", "pitch", "yaw", "throttle") if c in mc.columns]
    if not canais:
        return None, []
    base = np.column_stack([np.interp(t_grid, mc.index.to_numpy(),
                                      mc[c].to_numpy()) for c in canais])
    cols, nomes = [np.ones(len(t_grid))], ["1"]
    for L in lags_s:
        k = int(round(L * hz))
        for j, c in enumerate(canais):
            x = base[:, j]
            if k > 0:
                x = np.concatenate([np.full(k, x[0]), x[:-k]])
            cols.append(x)
            nomes.append(f"{c}(-{L:g}s)")
    return np.column_stack(cols), nomes


def dois_estagios(Xreg, y, Z):
    """2SLS. Devolve beta, R2 do primeiro estagio por regressor e R2 final."""
    Zq, _ = np.linalg.qr(Z)                      # base ortonormal do espaco de Z
    Xhat = Zq @ (Zq.T @ Xreg)
    r2_1 = []
    for j in range(Xreg.shape[1]):
        v = Xreg[:, j] - Xreg[:, j].mean()
        ss = float(np.sum(v ** 2))
        r2_1.append(float(1 - np.sum((Xreg[:, j] - Xhat[:, j]) ** 2) / ss)
                    if ss > 0 else np.nan)
    beta = np.linalg.lstsq(Xhat, y, rcond=None)[0]
    return beta, np.array(r2_1), r2(y, Xreg @ beta)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=10.0)
    ap.add_argument("--modes", nargs="*", default=["Manual"])
    ap.add_argument("--lag", type=float, default=0.06)
    ap.add_argument("--lags", type=float, nargs="*",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--ky", type=float, default=1.0)
    ap.add_argument("--kz", type=float, default=2.0)
    args = ap.parse_args(argv)

    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status", "manual_control_setpoint"])
    t, X, Xdot, U, g = sr.build_dataset(ulog, args.hz, args.fc, args.modes,
                                        None, None, "torque")
    U = sr.first_order_lag(U, args.lag, args.hz)
    nome = os.path.basename(args.ulg)[:8]

    Z, znomes = instrumentos(ulog, t, args.hz, args.lags)
    print("=" * 78)
    print(f"Variavel instrumental (2SLS) — {nome}")
    print(f"{len(t)} amostras @ {args.hz:g} Hz, modos={args.modes}")
    if Z is None:
        print("sem manual_control_setpoint utilizavel: 2SLS impossivel neste log")
        return 1
    print(f"instrumentos: {len(znomes)} colunas "
          f"({', '.join(znomes[1:5])}, ... atrasos {args.lags})")

    a = inertia_coefs(args.ky, args.kz)
    p, q, r_ = X[:, 0], X[:, 1], X[:, 2]
    T = U[:, 3]
    gyros = [q * r_, p * r_, p * q]
    estados = [p, q, r_]
    eixos = ["p_dot", "q_dot", "r_dot"]

    print("\nmodelo por eixo:  x_dot = a*gyro + b0*tau + b1*tau*T - c*estado")
    print(f"(coeficientes giroscopicos fixados em Iy/Ix={args.ky}, Iz/Ix={args.kz}: "
          f"{a.round(3)})\n")

    for i, nm in enumerate(eixos):
        y = Xdot[:, i] - a[i] * gyros[i]
        Xreg = np.column_stack([U[:, i], U[:, i] * T, -estados[i]])
        nomes = ["b0 (tau)", "b1 (tau*T)", "c (amortecimento)"]
        ols = np.linalg.lstsq(Xreg, y, rcond=None)[0]
        r2_ols = r2(y, Xreg @ ols)
        iv, r2_primeiro, r2_iv = dois_estagios(Xreg, y, Z)
        print(f"--- {nm}")
        print(f"    {'coeficiente':22} {'MQO':>12} {'2SLS':>12}   "
              f"{'forca do instrumento (R2 1o estagio)':>12}")
        for j, cn in enumerate(nomes):
            print(f"    {cn:22} {ols[j]:12.4g} {iv[j]:12.4g}   {r2_primeiro[j]:12.3f}")
        print(f"    R2 do ajuste: MQO {r2_ols:.3f} | 2SLS {r2_iv:.3f}")
        c_ols, c_iv = ols[2], iv[2]
        def leitura(c):
            if c > 1e-6:
                return f"amortecimento {c:+.3g} 1/s (fisico: >0 nesta convencao)"
            return f"amortecimento {c:+.3g} 1/s  <- NEGATIVO, nao fisico"
        print(f"    MQO : {leitura(c_ols)}")
        print(f"    2SLS: {leitura(c_iv)}")
        # b0 e b1 sao quase colineares (tau e tau*T): o que e identificavel e o
        # ganho efetivo no empuxo tipico, nao a divisao entre os dois termos
        Tm = float(np.mean(T))
        ge_ols = ols[0] + ols[1] * Tm
        ge_iv = iv[0] + iv[1] * Tm
        print(f"    ganho de controle efetivo em T medio ({Tm:.2f}): "
              f"MQO {ge_ols:.1f}  |  2SLS {ge_iv:.1f} rad/s² por unidade de tau "
              f"(diferenca {100*(ge_iv-ge_ols)/abs(ge_ols):+.0f}%)")
        print()

    # o instrumento e forte o bastante?
    print("--- diagnostico do instrumento")
    Xall = np.column_stack([U[:, 0], U[:, 1], U[:, 2], T])
    _, r2_1, _ = dois_estagios(Xall, Xdot[:, 0], Z)
    for nm, v in zip(["tau_x", "tau_y", "tau_z", "T"], r2_1):
        marca = ("forte" if v > 0.5 else "fraco" if v > 0.2 else "MUITO FRACO")
        print(f"    {nm:6}: R2 do primeiro estagio = {v:.3f}  ({marca})")
    print("\n    instrumento fraco => a estimativa 2SLS tem variancia alta e pode "
          "ser pior que o MQO; nesse caso o resultado nao corrige o vies, so o "
          "troca por ruido.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

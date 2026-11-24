"""Identificacao esparsa (SINDy) da dinamica de taxas angulares a partir de um log PX4.

Modelo-alvo (corpo rigido + atuacao + arrasto aerodinamico):
    p_dot = (Iy-Iz)/Ix * q*r + (1/Ix) * tau_x - (Kp/Ix) * p + ...
    q_dot = (Iz-Ix)/Iy * p*r + (1/Iy) * tau_y - (Kq/Iy) * q + ...
    r_dot = (Ix-Iy)/Iz * p*q + (1/Iz) * tau_z - (Kr/Iz) * r + ...

Os torques de comando tau_* sao reconstruidos a partir dos comandos de motor
(`actuator_motors`) e da geometria real do veiculo gravada no log (parametros
CA_ROTOR*_PX/PY/CT/KM), nao de um mixer presumido.

Uso:
  python src/sindy_rates.py data/ulg/<id>/<id>.ulg [--hz 50] [--modes Manual Acro]
                            [--t0 S --t1 S] [--fc 10] [--thresh 0.05]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402

R2D = 180.0 / np.pi


# ----------------------------------------------------------------- geometria
def rotor_geometry(ulog) -> dict:
    """Le CA_ROTOR*_{PX,PY,CT,KM} do log. Retorna matriz de alocacao normalizada."""
    par = ulog.initial_parameters
    n = int(par.get("CA_ROTOR_COUNT", 0))
    g = {"n": n, "px": [], "py": [], "ct": [], "km": []}
    for i in range(n):
        g["px"].append(float(par.get(f"CA_ROTOR{i}_PX", 0.0)))
        g["py"].append(float(par.get(f"CA_ROTOR{i}_PY", 0.0)))
        g["ct"].append(float(par.get(f"CA_ROTOR{i}_CT", 1.0)))
        g["km"].append(float(par.get(f"CA_ROTOR{i}_KM", 0.0)))
    for k in ("px", "py", "ct", "km"):
        g[k] = np.array(g[k])
    # arm "unitario" (PX/PY = +-1) significa geometria nao calibrada no veiculo
    g["normalized_arms"] = bool(np.allclose(np.abs(g["px"]), 1.0) and
                                np.allclose(np.abs(g["py"]), 1.0))
    return g


def virtual_torques(u: np.ndarray, g: dict) -> np.ndarray:
    """Comandos de motor (n_amostras x n_rotores) -> torques de comando [tx, ty, tz, T].

    F_i = CT_i * u_i * (0,0,-1);  tau_i = p_i x F_i + KM_i * CT_i * u_i * (0,0,-1)
      => tau_x = -CT * sum(py_i * u_i)
         tau_y = +CT * sum(px_i * u_i)
         tau_z = -CT * sum(km_i * u_i)
         T     = +CT * sum(u_i)     (empuxo total, modulo sinal)
    """
    ct, px, py, km = g["ct"], g["px"], g["py"], g["km"]
    tx = -(u * (ct * py)).sum(axis=1)
    ty = +(u * (ct * px)).sum(axis=1)
    tz = -(u * (ct * km)).sum(axis=1)
    T = +(u * ct).sum(axis=1)
    return np.column_stack([tx, ty, tz, T])


# -------------------------------------------------------------------- dados
def build_dataset(ulog, hz: float, fc: float, modes: list[str] | None,
                  t0: float | None, t1: float | None, source: str = "auto"):
    """source: 'torque' = vehicle_torque_setpoint (comando normalizado, taxa do
    laco de rate); 'motors' = actuator_motors reconstruido pela geometria CA_*."""
    rate = px4io.to_df(ulog, "vehicle_angular_velocity")[["xyz[0]", "xyz[1]", "xyz[2]"]]
    rate.columns = ["p", "q", "r"]
    g = rotor_geometry(ulog)
    present = {d.name for d in ulog.data_list}
    if source == "auto":
        source = "torque" if {"vehicle_torque_setpoint",
                              "vehicle_thrust_setpoint"} <= present else "motors"
    if source == "torque":
        tq = px4io.to_df(ulog, "vehicle_torque_setpoint")[["xyz[0]", "xyz[1]", "xyz[2]"]]
        tq.columns = ["tau_x", "tau_y", "tau_z"]
        th = px4io.to_df(ulog, "vehicle_thrust_setpoint")[["xyz[2]"]]
        th.columns = ["T"]
        ucols = ["tau_x", "tau_y", "tau_z", "T"]
        df = px4io.resample_join([rate, tq, th], hz)
    else:
        am = px4io.to_df(ulog, "actuator_motors")
        ucols = [f"control[{i}]" for i in range(g["n"])]
        am = am[ucols].fillna(0.0).clip(lower=0.0)
        df = px4io.resample_join([rate, am], hz)
    g["source"] = source

    # janela: armado, modos pedidos, recorte manual
    armed = px4io.armed_intervals(ulog)
    mask = np.zeros(len(df), dtype=bool)
    for a, b in armed:
        mask |= (df.index >= a) & (df.index <= b)
    if modes:
        mmask = np.zeros(len(df), dtype=bool)
        for a, b, name in px4io.mode_intervals(ulog):
            if name in modes:
                mmask |= (df.index >= a) & (df.index <= b)
        mask &= mmask
    if t0 is not None:
        mask &= df.index >= t0
    if t1 is not None:
        mask &= df.index <= t1

    # filtra e deriva no sinal continuo, depois aplica a mascara (evita bordas falsas)
    b, a = butter(4, fc / (hz / 2), btype="low")
    X = np.column_stack([filtfilt(b, a, df[c].to_numpy()) for c in ("p", "q", "r")])
    win = max(5, int(round(0.1 * hz)) | 1)  # ~100 ms, impar
    Xdot = np.column_stack([savgol_filter(X[:, i], win, 3, deriv=1, delta=1 / hz)
                            for i in range(3)])
    if g["source"] == "torque":
        U = df[ucols].to_numpy()
        U[:, 3] = -U[:, 3]  # thrust_setpoint z e negativo (FRD); usa modulo
    else:
        U = virtual_torques(df[ucols].to_numpy(), g)
    U = np.column_stack([filtfilt(b, a, U[:, i]) for i in range(U.shape[1])])

    # descarta bordas do filtro e trechos nao selecionados
    edge = int(2 * hz)
    mask[:edge] = False
    mask[-edge:] = False
    return df.index.to_numpy()[mask], X[mask], Xdot[mask], U[mask], g


# ---------------------------------------------------------------- biblioteca
def first_order_lag(u: np.ndarray, tau_s: float, hz: float) -> np.ndarray:
    """Atraso de primeira ordem (dinamica de motor) aplicado a cada coluna."""
    if tau_s <= 0:
        return u.copy()
    a = float(np.exp(-1.0 / (tau_s * hz)))
    y = np.empty_like(u)
    y[0] = u[0]
    for k in range(1, len(u)):
        y[k] = a * y[k - 1] + (1 - a) * u[k]
    return y


def library(X, U):
    p, q, r = X[:, 0], X[:, 1], X[:, 2]
    tx, ty, tz, T = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
    cols = {
        "1": np.ones_like(p),
        "p": p, "q": q, "r": r,
        "q*r": q * r, "p*r": p * r, "p*q": p * q,
        "tau_x": tx, "tau_y": ty, "tau_z": tz, "T": T,
        "p|p|": p * np.abs(p), "q|q|": q * np.abs(q), "r|r|": r * np.abs(r),
        # efetividade de controle cresce com o empuxo (rotor mais carregado)
        "tau_x*T": tx * T, "tau_y*T": ty * T, "tau_z*T": tz * T,
    }
    return np.column_stack(list(cols.values())), list(cols.keys())


def stlsq(Theta, y, threshold, n_iter=20, relative=True):
    """Sequential thresholded least squares com normalizacao de colunas.

    Com relative=True o limiar e uma fracao do maior coeficiente normalizado,
    ou seja, corta termos que contribuem menos que `threshold` vezes o termo
    dominante — independe das unidades do eixo.
    """
    s = np.linalg.norm(Theta, axis=0)
    s[s == 0] = 1.0
    Tn = Theta / s
    xi = np.linalg.lstsq(Tn, y, rcond=None)[0]
    if relative:
        threshold = threshold * np.max(np.abs(xi))
    abs_threshold = threshold
    active = np.ones(Tn.shape[1], dtype=bool)
    for _ in range(n_iter):
        small = np.abs(xi) < threshold
        if not small.any() or (~small).sum() == 0:
            break
        new_active = active & ~small
        if new_active.sum() == 0 or np.array_equal(new_active, active):
            break
        active = new_active
        xi = np.zeros(Tn.shape[1])
        xi[active] = np.linalg.lstsq(Tn[:, active], y, rcond=None)[0]
    return xi / s, abs_threshold


def r2(y, yhat):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else np.nan


def block_split(t, block_s=10.0):
    """Blocos alternados: pares = treino, impares = teste."""
    idx = ((t - t[0]) // block_s).astype(int)
    return idx % 2 == 0, idx % 2 == 1


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=None, help="corte do passa-baixa [Hz]")
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--thresh", type=float, default=None,
                    help="limiar relativo do STLSQ (fracao do termo dominante)")
    ap.add_argument("--input", choices=["auto", "torque", "motors"], default="auto",
                    help="origem do comando: torque setpoint ou comandos de motor")
    ap.add_argument("--lag", type=float, default=None,
                    help="constante de tempo do atuador [s]; omitido = varre")
    ap.add_argument("--cross-check", action="store_true",
                    help="confere os coeficientes contra o STLSQ do pysindy")
    args = ap.parse_args(argv)

    fc = args.fc if args.fc else min(15.0, 0.2 * args.hz)
    ulog = px4io.load(args.ulg, ["vehicle_angular_velocity", "actuator_motors",
                                 "vehicle_torque_setpoint", "vehicle_thrust_setpoint",
                                 "vehicle_status"])
    t, X, Xdot, U, g = build_dataset(ulog, args.hz, fc, args.modes, args.t0, args.t1,
                                     args.input)
    name = os.path.basename(args.ulg)[:8]

    print("=" * 76)
    print(f"SINDy — taxas angulares — {name} ({ulog.msg_info_dict.get('ver_hw','?')})")
    print(f"entrada = {g['source']}"
          + ("  (vehicle_torque_setpoint, normalizado -1..1, na taxa do laco de rate)"
             if g["source"] == "torque" else
             "  (actuator_motors + geometria CA_ROTOR*)"))
    print(f"rotores={g['n']}  CT={g['ct'][0]:.3g}  KM=±{abs(g['km'][0]):.3g}  "
          f"bracos {'NORMALIZADOS (+-1, geometria nao calibrada)' if g['normalized_arms'] else 'em metros'}")
    print(f"grade {args.hz:g} Hz, passa-baixa {fc:g} Hz, {len(t)} amostras "
          f"({len(t)/args.hz:.0f} s uteis)"
          + (f", modos={args.modes}" if args.modes else ""))
    print(f"excitacao: std(p,q,r) = "
          f"{np.std(X[:,0])*R2D:.1f}, {np.std(X[:,1])*R2D:.1f}, {np.std(X[:,2])*R2D:.1f} deg/s")

    tr, te = block_split(t)
    print(f"treino {tr.sum()} / teste {te.sum()} amostras (blocos alternados de 10 s)")

    # varredura da constante de tempo do atuador (motor + ESC)
    lags = [0.0] if args.lag is not None else [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2]
    if args.lag is not None:
        lags = [args.lag]
    best_lag, best_sum, scores = None, -np.inf, {}
    for L in lags:
        Ul = first_order_lag(U, L, args.hz)
        Th, _ = library(X, Ul)
        tot = 0.0
        for k in range(3):
            xi, _ = stlsq(Th[tr], Xdot[tr, k], 0.02)
            tot += r2(Xdot[te, k], Th[te] @ xi)
        scores[L] = tot / 3
        if tot > best_sum:
            best_sum, best_lag = tot, L
    if len(lags) > 1:
        print("atraso do atuador (s -> R2 medio): " +
              "  ".join(f"{L:g}->{scores[L]:.3f}" for L in lags))
    print(f"atraso adotado: {best_lag*1000:.0f} ms")
    U = first_order_lag(U, best_lag, args.hz)
    Theta, names = library(X, U)

    thresholds = ([args.thresh] if args.thresh else
                  [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35])
    axes = ["p_dot", "q_dot", "r_dot"]
    results = {}
    for k, ax_name in enumerate(axes):
        y = Xdot[:, k]
        cand = []
        for th in thresholds:
            xi, abs_th = stlsq(Theta[tr], y[tr], th)
            sc = r2(y[te], Theta[te] @ xi)
            cand.append((sc, th, xi, int(np.sum(np.abs(xi) > 0)), abs_th))
        # parcimonia: menos termos entre os que ficam a <=3% do melhor R2
        best_r2 = max(c[0] for c in cand)
        ok = [c for c in cand if c[0] >= best_r2 - 0.03 * abs(best_r2)]
        sc, th, xi, nterm, abs_th = min(ok, key=lambda c: (c[3], -c[0]))
        results[ax_name] = (xi, names, sc, th, abs_th)
        if len(thresholds) > 1:
            print("    varredura (limiar -> termos, R2 teste): " +
                  "  ".join(f"{c[1]:g}->{c[3]}/{c[0]:.3f}" for c in cand))
        print(f"\n--- {ax_name}   R2(teste) = {sc:.3f}   limiar = {th:g}   "
              f"termos = {int(np.sum(np.abs(xi) > 0))}")
        print(f"    R2(treino) = {r2(y[tr], Theta[tr] @ xi):.3f}   "
              f"RMS residuo = {np.std(y - Theta @ xi)*R2D:.1f} deg/s^2")
        order = np.argsort(-np.abs(xi * np.std(Theta, axis=0)))
        for i in order:
            if xi[i] == 0:
                continue
            contrib = np.std(Theta[:, i] * xi[i]) * R2D
            print(f"      {names[i]:>8}  coef = {xi[i]:+12.4g}   "
                  f"contribuicao std = {contrib:7.1f} deg/s^2")

    if args.cross_check:
        import pysindy as ps
        print("\n--- conferencia com pysindy.STLSQ (mesma biblioteca, mesmo limiar)")
        for k, ax_name in enumerate(axes):
            xi, _, _, th, abs_th = results[ax_name]
            opt = ps.STLSQ(threshold=abs_th, normalize_columns=True, alpha=0.0)
            opt.fit(Theta[tr], Xdot[tr, k])
            xp = opt.coef_.ravel()
            d = np.max(np.abs(xp - xi)) / max(np.max(np.abs(xi)), 1e-12)
            print(f"    {ax_name}: diferenca relativa max = {d:.2e} "
                  f"({'igual' if d < 1e-6 else 'DIVERGE'})")

    # ------------------------------------------------- interpretacao fisica
    print("\n--- leitura fisica (corpo rigido)")
    gyro_terms = {"p_dot": "q*r", "q_dot": "p*r", "r_dot": "p*q"}
    ratio_name = {"p_dot": "(Iy-Iz)/Ix", "q_dot": "(Iz-Ix)/Iy", "r_dot": "(Ix-Iy)/Iz"}
    tau_term = {"p_dot": "tau_x", "q_dot": "tau_y", "r_dot": "tau_z"}
    damp = {"p_dot": "p", "q_dot": "q", "r_dot": "r"}
    for ax_name in axes:
        xi, nm, sc, th = results[ax_name][:4]
        d = dict(zip(nm, xi))
        gy, ta, dp = gyro_terms[ax_name], tau_term[ax_name], damp[ax_name]
        print(f"  {ax_name}:")
        print(f"    {ratio_name[ax_name]:>12} ~ {d.get(gy,0.0):+9.4g}  (coef de {gy})")
        if d.get(ta, 0.0) != 0:
            print(f"    ganho de controle (1/I)*dtau: {d[ta]:+9.4g} rad/s^2 por unidade de {ta}")
        if d.get(dp, 0.0) != 0:
            tau_c = -1.0 / d[dp] if d[dp] < 0 else np.nan
            print(f"    amortecimento: {d[dp]:+9.4g} 1/s"
                  + (f"  -> constante de tempo {tau_c:.2f} s" if np.isfinite(tau_c) else
                     "  (POSITIVO: instavel/nao fisico, provavel acoplamento com o controlador)"))
    print()
    np.save(os.path.join("out", f"sindy_{name}.npy"),
            {"names": names, "coefs": {k: v[0] for k, v in results.items()},
             "r2": {k: v[2] for k, v in results.items()}, "lag_s": best_lag,
             "hz": args.hz, "fc": fc, "t": t, "X": X, "Xdot": Xdot, "U": U,
             "train": tr, "test": te, "source": g["source"]}, allow_pickle=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

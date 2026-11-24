"""Valida graficamente o modelo identificado por src/sindy_rates.py.

Esquerda: derivada medida vs prevista num trecho de teste.
Direita:  simulacao do modelo em horizonte curto (reinicia na medida a cada H s).

Uso: python src/plot_sindy.py out/sindy_<id>.npy [--win 20] [--horizon 2]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sindy_rates as sr  # noqa: E402

C_MEAS = "#1F6FB2"
C_MODEL = "#D1670B"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e0"
R2D = 180.0 / np.pi


def simulate(X, U, coefs, names, hz, horizon_s, t):
    """Integra o modelo (Euler) em janelas de horizon_s, reiniciando na medida."""
    n = len(t)
    H = int(horizon_s * hz)
    sim = np.empty_like(X)
    idx = {k: i for i, k in enumerate(names)}
    C = np.column_stack([coefs["p_dot"], coefs["q_dot"], coefs["r_dot"]])
    for s0 in range(0, n, H):
        x = X[s0].copy()
        for k in range(s0, min(s0 + H, n)):
            sim[k] = x
            th = sr.library(x[None, :], U[k][None, :])[0][0]
            x = x + (th @ C) / hz
    return sim


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("--win", type=float, default=20.0, help="janela mostrada [s]")
    ap.add_argument("--horizon", type=float, default=2.0)
    args = ap.parse_args(argv)

    d = np.load(args.npy, allow_pickle=True).item()
    t, X, Xdot, U = d["t"], d["X"], d["Xdot"], d["U"]
    names, hz, te = d["names"], d["hz"], d["test"]
    axes_n = ["p_dot", "q_dot", "r_dot"]
    lbl = ["p", "q", "r"]

    Theta, _ = sr.library(X, U)
    pred = {a: Theta @ d["coefs"][a] for a in axes_n}
    sim = simulate(X, U, d["coefs"], names, hz, args.horizon, t)

    # janela de teste mais movimentada
    te_idx = np.where(te)[0]
    w = int(args.win * hz)
    best, bi = -1, te_idx[0]
    for s0 in range(te_idx[0], te_idx[-1] - w, w // 2):
        sl = slice(s0, s0 + w)
        if not te[sl].all():
            continue
        e = np.std(X[sl, 0]) + np.std(X[sl, 1])
        if e > best:
            best, bi = e, s0
    sl = slice(bi, bi + w)

    fig, axs = plt.subplots(3, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#fcfcfb")
    for k in range(3):
        a0, a1 = axs[k, 0], axs[k, 1]
        a0.plot(t[sl], Xdot[sl, k] * R2D, color=C_MEAS, lw=1.3, label="medido")
        a0.plot(t[sl], pred[axes_n[k]][sl] * R2D, color=C_MODEL, lw=1.1, ls="--",
                label="modelo")
        a0.set_ylabel(f"d{lbl[k]}/dt [deg/s²]", fontsize=9, color=INK)
        a0.text(0.995, 0.04, f"R² teste = {d['r2'][axes_n[k]]:.3f}",
                transform=a0.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)

        a1.plot(t[sl], X[sl, k] * R2D, color=C_MEAS, lw=1.3, label="medido")
        a1.plot(t[sl], sim[sl, k] * R2D, color=C_MODEL, lw=1.1, ls="--",
                label=f"simulado ({args.horizon:g} s)")
        a1.set_ylabel(f"{lbl[k]} [deg/s]", fontsize=9, color=INK)
        err = np.sqrt(np.mean((sim[te, k] - X[te, k]) ** 2)) * R2D
        a1.text(0.995, 0.04, f"RMS horizonte {args.horizon:g}s = {err:.1f} deg/s",
                transform=a1.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)

        for ax in (a0, a1):
            ax.set_facecolor("#fcfcfb")
            ax.grid(True, color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for s in ax.spines.values():
                s.set_color(GRID)
            if k == 0:
                ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    axs[2, 0].set_xlabel("tempo [s]", fontsize=9, color=INK)
    axs[2, 1].set_xlabel("tempo [s]", fontsize=9, color=INK)
    name = os.path.basename(args.npy).replace("sindy_", "").replace(".npy", "")
    fig.suptitle(f"Modelo identificado — {name}  "
                 f"(entrada {d['source']}, atraso {d['lag_s']*1000:.0f} ms, "
                 f"grade {hz:g} Hz) — trecho de teste", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join("out", f"sindy_val_{name}.png")
    fig.savefig(out, dpi=130)
    print("figura:", out)
    for k in range(3):
        e = np.sqrt(np.mean((sim[te, k] - X[te, k]) ** 2)) * R2D
        print(f"  {lbl[k]}: RMS simulacao {args.horizon:g}s = {e:6.2f} deg/s   "
              f"(std do sinal = {np.std(X[te,k])*R2D:6.2f} deg/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

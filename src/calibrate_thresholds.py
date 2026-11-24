"""Calibra os limiares do relatorio contra uma amostra da frota publica.

Os limiares de `src/flight_report.py` foram escolhidos por mim, no vacuo. Este
script baixa uma amostra de logs publicos (um por veiculo, para nao enviesar),
mede as mesmas grandezas que o relatorio mede, apaga o log em seguida — o disco
nao cresce — e imprime a distribuicao empirica de cada metrica.

Com isso o limiar deixa de ser chute e passa a ser percentil: "pior que 90% da
frota publica" e uma afirmacao verificavel.

Uso:
  python src/calibrate_thresholds.py ids.txt [--max 220] [--out out/calibracao.json]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import px4io  # noqa: E402
import flight_report as fr  # noqa: E402

URL = "https://cdn.logs.px4.io/{}.ulg"


def baixar(log_id: str, destino: str, limite_bytes: int = 60_000_000) -> str | None:
    alvo = os.path.join(destino, f"{log_id}.ulg")
    r = subprocess.run(["curl", "-sSL", "-m", "240", "--max-filesize",
                        str(limite_bytes), "-o", alvo, URL.format(log_id)],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(alvo) or os.path.getsize(alvo) < 10000:
        return None
    return alvo


def medir(path: str) -> dict | None:
    """Mede, com as MESMAS funcoes do relatorio, as grandezas que viram limiar."""
    u = px4io.load(path)
    ctx = {"u": u, "armed": px4io.armed_intervals(u),
           "modos": px4io.mode_intervals(u)}
    _, d_att = fr.check_atitude(u, ctx)
    _, d_vib = fr.check_vibracao(u, ctx)
    _, d_ekf = fr.check_estimador(u, ctx)
    _, d_act = fr.check_atuacao(u, ctx)
    _, d_bat = fr.check_bateria(u, ctx)
    m = fr.medidas(u, d_att, d_vib, d_ekf, d_act, d_bat)
    return m or None


def resumo(valores: dict, saida: str):
    print("\n" + "=" * 92)
    print(f"{'metrica':16} {'n':>4} {'p10':>9} {'p50':>9} {'p75':>9} {'p90':>9} "
          f"{'p95':>9} {'p99':>9} {'max':>9}")
    print("-" * 92)
    tabela = {}
    for k, v in sorted(valores.items()):
        v = np.array([x for x in v if np.isfinite(x)])
        if v.size < 10:
            continue
        q = np.percentile(v, [10, 50, 75, 90, 95, 99])
        tabela[k] = dict(n=int(v.size), p10=q[0], p50=q[1], p75=q[2], p90=q[3],
                         p95=q[4], p99=q[5], max=float(v.max()))
        print(f"{k:16} {v.size:4} {q[0]:9.3g} {q[1]:9.3g} {q[2]:9.3g} {q[3]:9.3g} "
              f"{q[4]:9.3g} {q[5]:9.3g} {v.max():9.3g}")
    json.dump(tabela, open(saida, "w"), indent=1)
    json.dump({k: list(map(float, v)) for k, v in valores.items()},
              open(saida.replace(".json", "_cru.json"), "w"))
    print(f"\ndistribuicoes salvas em {saida}")

    # sugestao: ATENCAO no p90, FALHA no p99 (metricas em que "maior = pior")
    maior_pior = {"att_roll": "att_auto_warn_deg/att_auto_fail_deg",
                  "rate_p": "rate_auto_warn_dps/rate_auto_fail_dps",
                  "hf_accel": "hf_accel_warn/hf_accel_fail",
                  "vib_metric": "vib_warn/vib_fail",
                  "innov_max": "innov_warn/innov_fail",
                  "sat_pct": "sat_warn_pct/sat_fail_pct",
                  "desbal_pct": "desbal_warn_pct/desbal_fail_pct",
                  "dropout_s": "dropout_warn_s/dropout_fail_s"}
    print("\n--- limiares sugeridos (ATENCAO = p75 da frota, FALHA = p90)")
    print(f"{'limiar em LIMIARES':42} {'atual':>14} {'sugerido':>20}")
    atual = fr.LIMIARES
    for met, nomes in maior_pior.items():
        if met not in tabela:
            continue
        n_w, n_f = nomes.split("/")
        print(f"{nomes:42} {atual.get(n_w,'?'):>7}/{atual.get(n_f,'?'):<6} "
              f"{tabela[met]['p75']:>9.3g}/{tabela[met]['p90']:<9.3g}")
    if "vcel_min" in tabela:
        print(f"{'cell_warn_v/cell_fail_v (menor = pior)':42} "
              f"{atual['cell_warn_v']:>7}/{atual['cell_fail_v']:<6} "
              f"{np.percentile([x for x in valores['vcel_min'] if np.isfinite(x)], 25):>9.3g}/"
              f"{tabela['vcel_min']['p10']:<9.3g}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("--max", type=int, default=220)
    ap.add_argument("--out", default="out/calibracao.json")
    args = ap.parse_args(argv)

    ids = [l.strip() for l in open(args.ids) if l.strip()][:args.max]
    tmp = tempfile.mkdtemp(prefix="calib_ulg_")
    valores: dict[str, list] = {}
    ok = falha = 0
    try:
        for i, log_id in enumerate(ids, 1):
            p = baixar(log_id, tmp)
            if p is None:
                falha += 1
                print(f"[{i:3}/{len(ids)}] {log_id[:8]} download falhou", flush=True)
                continue
            try:
                m = medir(p)
            except Exception as e:
                m = None
                print(f"[{i:3}/{len(ids)}] {log_id[:8]} erro: {str(e)[:60]}", flush=True)
            finally:
                os.remove(p)
            if not m:
                falha += 1
                continue
            ok += 1
            for k, v in m.items():
                valores.setdefault(k, []).append(v)
            print(f"[{i:3}/{len(ids)}] {log_id[:8]} ok "
                  f"({len(m)} metricas, acumulado {ok})", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nlogs medidos: {ok} | falhas: {falha}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    resumo(valores, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

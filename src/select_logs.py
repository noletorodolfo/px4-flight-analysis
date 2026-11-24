"""Seleciona logs candidatos no banco publico da PX4 Flight Review.

Baixe o indice antes:
  curl -sSL https://review.px4.io/dbinfo | gunzip > dbinfo.json
Uso: python src/select_logs.py dbinfo.json
Gera candidates.json no diretorio atual.
"""
import json, sys

MODES = {0:"Manual",1:"Altitude",2:"Position",3:"Mission",4:"Hold",5:"Return",6:"Acro",
         7:"Offboard",8:"Stabilized",10:"Takeoff",11:"Land",12:"Descend",13:"FollowMe",
         14:"PrecLand",15:"Orbit"}
MANUAL = {0,6,8}  # manobras com piloto no loop

path = sys.argv[1]
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    s = f.read()

dec = json.JSONDecoder()
i = s.index('[') + 1
n = len(s)
out = []
total = 0
while i < n:
    while i < n and s[i] in ' \t\r\n,':
        i += 1
    if i >= n or s[i] == ']':
        break
    obj, j = dec.raw_decode(s, i)
    i = j
    total += 1
    d = obj.get('duration_s') or 0
    mav = obj.get('mav_type') or ''
    if 'rotor' not in mav.lower() and 'copter' not in mav.lower():
        continue
    if not (150 <= d <= 1200):
        continue
    if obj.get('num_logged_errors', 0) > 0:
        continue
    rel = obj.get('ver_sw_release') or ''
    fmd = obj.get('flight_mode_durations') or []
    manual_s = sum(t for m, t in fmd if m in MANUAL)
    nmodes = len(set(m for m, t in fmd))
    if manual_s < 40 or nmodes < 2:
        continue
    out.append({
        'log_id': obj['log_id'], 'date': obj.get('log_date'), 'dur': d,
        'mav': mav, 'rel': rel, 'hw': obj.get('sys_hw'),
        'airframe': obj.get('airframe_name'), 'manual_s': manual_s,
        'modes': [[MODES.get(m, m), t] for m, t in fmd],
        'url': obj.get('download_url'), 'desc': (obj.get('description') or '')[:60],
    })

print(f"total logs no banco: {total}; candidatos: {len(out)}")
out.sort(key=lambda r: (r['date'] or '', r['manual_s']), reverse=True)
json.dump(out, open('candidates.json','w'), indent=1)
for r in out[:25]:
    print(f"{r['date']} {r['dur']:5}s manual={r['manual_s']:4}s {r['mav']:12} {str(r['rel'])[:12]:12} {str(r['hw'])[:18]:18} {r['log_id']}")
    print(f"    modos={r['modes']} | {r['desc']}")

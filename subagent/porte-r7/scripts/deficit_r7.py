"""Por bloco do recorde 7: conteúdo (unidades da grade; 2 = 1 nível) de cada trecho do caminho principal entre
refreshes, a folga não usada de cada trecho e quantos níveis faltam para tirar UM refresh do trecho kv -> up.
Trechos: B = pouso do push K/V -> entrada do SB#1; C = pouso do SB#1 -> entrada do SB#2; D = pouso do SB#2 -> entrada
do refresh denso (up / ln_2 / down); A = pouso do denso -> entrada do push K/V do bloco seguinte.
Conteúdo = entrada do refresh seguinte - pouso do anterior (o que as ops consomem). Tetos: entrada <= CAP (52 densa,
52 ou 54 esparsa); pouso mínimo: StC-first 26 (drop 0), SparseB-256 24 (drop base 1).
Uso: python3 -I deficit_r7.py PROGS"""
import json, sys
progs = sys.argv[1]
CAP_D, LAND_STCF, LAND_SB = 52, 26, 24
rows = []
for b in range(12):
    s = json.load(open(f"{progs}/block_{b}.json"))["bts_sites"]
    main = [x for x in s if x["route"]["slots"] != 1]          # sem os SPRU (estatísticas da LN, ramo lateral)
    kv = next(x for x in main if x["kind"] == "Deliberate")
    sbs = [x for x in main if x["route"]["slots"] == 256]
    dense = [x for x in main if x["route"]["slots"] == 0 and x is not kv]
    nxt = json.load(open(f"{progs}/block_{b + 1}.json"))["bts_sites"] if b < 11 else None
    kv_next = next((x for x in nxt if x["kind"] == "Deliberate"), None) if nxt else None
    sb1, sb2 = sbs[0], sbs[-1]
    up = dense[0]                      # o primeiro denso depois do SB#2 (blocos 2 e 11: ln_2; o 2º denso fica em A)
    last = dense[-1]
    B = sb1["input"]["level"] - kv["landing"]["level"]
    C = sb2["input"]["level"] - sb1["landing"]["level"]
    D = up["input"]["level"] - sb2["landing"]["level"]
    A = (kv_next["input"]["level"] - last["landing"]["level"]) if kv_next else None
    slack = dict(B=CAP_D - sb1["input"]["level"], C=CAP_D - sb2["input"]["level"], D=CAP_D - up["input"]["level"],
                 A=(CAP_D - kv_next["input"]["level"]) if kv_next else None)
    if b == 11:                        # bloco 11: um SB-512 no qkt entre o push K/V e o SB#1 (B conta os dois trechos)
        pass
    tot = B + C + D
    for cap_sb in (52, 54):
        room = (cap_sb - LAND_STCF) + (CAP_D - LAND_SB)
        rows.append((b, cap_sb, B, C, D, A, tot, room, tot - room, slack,
                     [x["route"]["raise_drop"] for x in (sb1, sb2, up)], len(sbs)))
print("bloco capSB |  B  C  D  A | B+C+D  sala  falta(grade) falta(níveis) | folga não usada (níveis) B/C/D/A | drops SB1/SB2/denso")
for b, cap, B, C, D, A, tot, room, miss, sl, drops, nsb in rows:
    f = lambda v: "-" if v is None else f"{v / 2:.0f}"
    print(f"{b:5d} {cap:5d} | {B:2d} {C:2d} {D:2d} {A if A is not None else '-':>2} | {tot:5d} {room:5d} {miss:8d} {miss / 2:10.1f}    | "
          f"{f(sl['B'])}/{f(sl['C'])}/{f(sl['D'])}/{f(sl['A'])} | {drops} ({nsb} SB-256)")

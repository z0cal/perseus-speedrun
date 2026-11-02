"""Como deficit_r7.py, mas com o fluxo residual: depois de tirar um SparseB, o ramo de atenção chega ao x_mid mais
fundo, e x_mid é a saída do bloco (x_out = max(down, x_mid), residual_r7.log) que alimenta o ln_1 -> qkv -> push K/V
do bloco seguinte. Teto do x_mid = nível atual da saída + a folga do push K/V seguinte (entrada <= 52).
Conteúdo = B + C + (x_mid - pouso do SB#2). Sala = (capSB - 26) + (teto x_mid - 24).
Uso: python3 -I deficit_stream_r7.py PROGS GRAPHS_GNABM"""
import json, sys
progs, gd = sys.argv[1:3]
def lvl(d, nm):
    V = d["vars"]
    ids = [i for i, v in enumerate(V) if v["name"] == nm or v["name"].startswith(nm + "~")]
    return max(V[i]["meta"]["level"] for i in ids)
print("bloco | B  C  Dattn | conteúdo | teto x_mid | falta (níveis) capSB 52 / 54")
for b in range(11):
    d = json.load(open(f"{progs}/block_{b}.json"))
    g = json.load(open(f"{gd}/block_{b}/graph.json"))["nodes"]
    s = [x for x in d["bts_sites"] if x["route"]["slots"] != 1]
    kv = next(x for x in s if x["kind"] == "Deliberate")
    sbs = [x for x in s if x["route"]["slots"] == 256]
    sb1, sb2 = sbs[0], sbs[-1]
    adds = [n["output"] for n in g if n["step"] == f"blk{b}" and n["op_type"] == "add"]
    Lmid, Lexit = lvl(d, adds[0]), lvl(d, adds[1])
    dn = json.load(open(f"{progs}/block_{b + 1}.json"))
    kvn = next(x for x in dn["bts_sites"] if x["kind"] == "Deliberate")
    cap_mid = Lexit + (52 - kvn["input"]["level"])     # x_mid pode descer até a saída do bloco + a folga do push seguinte
    B = sb1["input"]["level"] - kv["landing"]["level"]
    C = sb2["input"]["level"] - sb1["landing"]["level"]
    Da = Lmid - sb2["landing"]["level"]
    tot = B + C + Da
    miss = [(tot - ((cs - 26) + (cap_mid - 24))) / 2 for cs in (52, 54)]
    print(f"{b:5d} | {B:2d} {C:2d} {Da:3d}   | {tot:6d}   | {cap_mid:6d}     | {miss[0]:4.1f} / {miss[1]:4.1f}")

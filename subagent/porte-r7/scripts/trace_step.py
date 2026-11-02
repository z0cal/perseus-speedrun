"""Traço de um passo do grafo com os níveis PLANEJADOS (programa compilado do recorde 7): op, saída, nível, entradas.
Uso: python3 -I trace_step.py PROGS GRAPHS BLOCO PASSO[,PASSO...]"""
import json, sys
progs, gd, b, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")
d = json.load(open(f"{progs}/block_{b}.json"))
g = json.load(open(f"{gd}/block_{b}/graph.json"))["nodes"]
V, N = d["vars"], d["names"]
bts = {s["var"]: s for s in d["bts_sites"]}
lvl = lambda nm: V[N[nm]]["meta"]["level"] if nm in N else "-"
for n in g:
    if n["step"].split(".")[-1] not in steps:
        continue
    o = n["output"]
    mark = ""
    if o in bts:
        s = bts[o]; mark = f"  <== REFRESH {s['route']['slots']} in {s['input']['level']} -> {s['landing']['level']}"
    ins = ",".join(f"{x}@{lvl(x)}" for x in n["inputs"][:3])
    print(f"{n['step']:16s} {n['op_type']:14s} {o:22s} L{lvl(o)!s:3s} max|x| {n.get('output_max_abs', 0):.3g}  <- {ins}{mark}")

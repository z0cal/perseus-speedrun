"""Lista os refreshes do plano do recorde 7 por bloco, com o passo (step) do grafo, rota, nível de entrada e pouso,
e o intervalo de níveis planejados de cada passo. Uso: python3 -I sites_r7.py PROGS GRAPHS_GNABM GRAPHS_GN_CM"""
import json, sys, collections
progs, g012, g13 = sys.argv[1:4]
def route(r):
    if r["slots"] == 0:
        return ("StCf" if r["stc_first"] else "dense") + ("R" if r["real"] else "C")
    if r["slots"] == 1:
        return "SPRU"
    return f"SB{r['slots']}"
tot = collections.Counter()
for b in range(14):
    d = json.load(open(f"{progs}/block_{b}.json"))
    g = json.load(open(f"{g13 if b == 13 else g012}/block_{b}/graph.json"))["nodes"]
    step = {n["output"]: n["step"] for n in g}
    V, N = d["vars"], d["names"]
    lvl = lambda nm: V[N[nm]]["meta"]["level"] if nm in N else None
    print(f"== bloco {b}: {len(d['bts_sites'])} refreshes")
    for s in d["bts_sites"]:
        v = s["var"]; st = step.get(v, step.get(v.replace("_auto_bootstrapped", ""), "?"))
        r = s["route"]; rt = route(r); tot[rt] += 1
        print(f"   {v:22s} {st:18s} {rt:7s} {s['kind']:10s} in {s['input']['level']:2d} -> pouso {s['landing']['level']:2d} drop {r['raise_drop']} cf {r['cf']}")
    rng = collections.OrderedDict()
    for n in g:
        L = lvl(n["output"])
        if L is None: continue
        a = rng.setdefault(n["step"], [99, 0]); a[0] = min(a[0], L); a[1] = max(a[1], L)
    print("   níveis por passo: " + ", ".join(f"{k.split('.')[-1]} {a[0]}-{a[1]}" for k, a in rng.items()))
print(dict(tot), sum(tot.values()))

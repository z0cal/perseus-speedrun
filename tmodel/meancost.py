"""Per LN site: the ms of the mean path (ops after the centring mask, before the variance square). usage: python -I meancost.py PROGS PLAN GRAPH"""
import json, os, re, sys
sys.path.insert(0, "/home/zocal/pesquisa/perseus-speedrun/tmodel")
import tokplan as T, units as U  # noqa: E402
C = json.load(open("/home/zocal/pesquisa/perseus-speedrun/tmodel/calib.json"))
ms = lambda u: C["tau"] * u[0] + C["kappa"] * u[1] + C["eps"] * u[2]  # noqa: E731
prog, plan, gdir = sys.argv[1:4]
tot = 0
for b in range(14):
    f = f"{prog}/{plan}/block_{b}.json"
    if not os.path.exists(f) or not os.path.exists(f"{gdir}/block_{b}/graph.json"):
        continue
    p = json.load(open(f)); g = json.load(open(f"{gdir}/block_{b}/graph.json"))["nodes"]
    V = p["vars"]; lv = lambda i: V[i]["meta"]["level"]  # noqa: E731
    steps = sorted({n.get("step") for n in g if n.get("step") and re.search(r"ln", n.get("step"))})
    for st in steps:
        ops = [o for o in p["ops"] if "Node" in (o.get("origin") or {}) and g[o["origin"]["Node"]].get("step") == st]
        kinds = [o["kind"] if isinstance(o["kind"], str) else next(iter(o["kind"])) for o in ops]
        if "Square" not in kinds or "MultPt" not in kinds:
            continue
        i0, i1 = kinds.index("MultPt"), kinds.index("Square")
        t = 0
        for o in ops[i0 + 1:i1]:
            acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
            T.op_units(o, lv, acc)
            t += sum(ms(acc[c]) for c in ("ks", "rescale", "ew"))
        tot += t
        print(f"{st:14s} mean path {t:5.2f} ms  ops {kinds[i0 + 1:i1]}")
print("total mean path ms/token", round(tot, 2))

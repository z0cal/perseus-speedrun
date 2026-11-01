"""Op time (calibrated units) per model step (graph node `step`, block prefix stripped) and its limb profile.
usage: python -I steps.py PROGS_DIR PLAN GRAPH_DIR"""
import collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokplan as T, units as U  # noqa: E402
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")))
ms = lambda u: C["tau"] * u[0] + C["kappa"] * u[1] + C["eps"] * u[2]  # noqa: E731
prog, plan, gdir = sys.argv[1:4]
by = collections.defaultdict(lambda: [0.0, 0.0, collections.Counter()])
for b in range(14):
    p = json.load(open(f"{prog}/{plan}/block_{b}.json"))
    g = json.load(open(f"{gdir}/block_{b}/graph.json"))["nodes"]
    V = p["vars"]
    lv = lambda i: V[i]["meta"]["level"]  # noqa: E731
    def step_of(o):
        org = o.get("origin") or {}
        if "Node" in org and org["Node"] < len(g):
            return re.sub(r"^blk\d+\.", "", g[org["Node"]].get("step") or "?")
        return "(" + next(iter(org), "?") + ")"
    def visit(o):
        if T.kind(o) == "Batched":
            for s in o["kind"]["Batched"]:
                visit(s)
            return
        acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
        T.op_units(o, lv, acc)
        st = step_of(o) if b < 13 else "stage13." + step_of(o)
        e = by[st]
        e[0] += ms(acc["ks"]); e[1] += ms(acc["rescale"]) + ms(acc["ew"])
        outs = [lv(i) for i in o["outs"]] or [lv(i) for i in o["ins"]] or [0]
        if T.kind(o) in ("Rotate", "RotateSum", "Conjugate", "Mult", "Square"):
            e[2][T.NQ - max(outs)] += 1
    for o in p["ops"]:
        visit(o)
tot = sum(v[0] + v[1] for v in by.values())
print(f"ops total {tot:.1f} ms")
for k, (ks, rest, ns) in sorted(by.items(), key=lambda x: -(x[1][0] + x[1][1]))[:30]:
    print(f"  {k:28s} {ks + rest:6.1f} ms (KS {ks:5.1f})  KS ops by limbs: {dict(sorted(ns.items()))}")

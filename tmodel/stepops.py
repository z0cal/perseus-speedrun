"""One block's ops of a step, in order: kind, limbs, ms. usage: python -I stepops.py PROGS PLAN GRAPH BLOCK STEP"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokplan as T, units as U  # noqa: E402
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")))
ms = lambda u: C["tau"] * u[0] + C["kappa"] * u[1] + C["eps"] * u[2]  # noqa: E731
prog, plan, gdir, b, step = sys.argv[1:6]
p = json.load(open(f"{prog}/{plan}/block_{b}.json")); g = json.load(open(f"{gdir}/block_{b}/graph.json"))["nodes"]
V = p["vars"]; lv = lambda i: V[i]["meta"]["level"]  # noqa: E731
tot = 0
for o in p["ops"]:
    org = o.get("origin") or {}
    if "Node" not in org or not re.search(step, g[org["Node"]].get("step") or ""):
        continue
    acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
    T.op_units(o, lv, acc)
    t = sum(ms(acc[c]) for c in ("ks", "rescale", "ew")); tot += t
    k = o["kind"] if isinstance(o["kind"], str) else json.dumps(o["kind"])[:60]
    nd = g[org["Node"]]
    print(f"{t:6.3f} ms  n={T.NQ - max([lv(i) for i in o["outs"]] or [lv(i) for i in o["ins"]] or [99])}  {k:60s} pre={o['pre']}  {nd['op_type']} {nd['output']}")
print("total", round(tot, 2))

"""Per refresh of a plan: its route, the step that produced the refreshed value and the steps of its consumers.
usage: python -I refresh_ctx.py PLAN_DIR GRAPH_DIR BLOCK"""
import json, sys
from collections import defaultdict

plan, graph, b = sys.argv[1:]
p = json.load(open(f"{plan}/block_{b}_placement.json"))
g = json.load(open(f"{graph}/block_{b}/graph.json"))["nodes"]
prod = {n["output"]: n for n in g}
cons = defaultdict(list)
for n in g:
    for x in n["inputs"]:
        cons[x].append(n)
sites = [(pl["target_var"], "placed") for pl in p["placements"]] + [(v, "hint") for v in p["hint_fire"]]
sites += [(n["output"], "deliberate") for n in g if n["op_type"] == "deliberate_bootstrap"]
for v, kind in sites:
    sl = p["sparse_slots"].get(v)
    route = f"sparse{sl}" if sl else ("dense(real)" if v in p.get("real_route", []) else "dense")
    src = prod.get(v, {})
    nxt = sorted({c["step"] for c in cons[v]})
    print(f"{v:7s} {kind:10s} {route:12s} cf={p['correction_factor'].get(v, '-'):>3} "
          f"from {src.get('step', '?')}/{src.get('op_type', '?'):18s} -> {', '.join(nxt)[:90]}")

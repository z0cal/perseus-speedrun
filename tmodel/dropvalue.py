"""Upper bound of one more raise drop per refresh site: the refresh's own saving (units at 2 fewer limbs) plus every op
fed by its output, up to the next refresh, at 2 fewer limbs. Feasibility ignored (the planner decides that).
usage: python -I dropvalue.py PROGS PLAN"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokplan as T, units as U, predict as P  # noqa: E402
ms = P.ms
d = sys.argv[1] + "/" + sys.argv[2]
_, sites_all, cfg = T.plan_units(d)
sd, ss = T.series_of(cfg)
rows = []
for b in range(14):
    p = json.load(open(f"{d}/block_{b}.json")); V = p["vars"]
    ops = []
    def flat(o):
        if T.kind(o) == "Batched":
            for s in o["kind"]["Batched"]:
                flat(s)
        else:
            ops.append(o)
    for o in p["ops"]:
        flat(o)
    consumers = collections.defaultdict(list)
    for i, o in enumerate(ops):
        for v in o["ins"]:
            consumers[v].append(i)
    def cost(o, shift):
        acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
        T.op_units(o, lambda i: V[i]["meta"]["level"] + shift, acc)
        return sum(ms(acc[c]) for c in ("ks", "rescale", "ew"))
    name_of = {i: v["name"] for i, v in enumerate(V)}
    for o in ops:
        if T.kind(o) != "Bootstrap":
            continue
        out = o["outs"][0]
        seen, stack, dops = set(), [out], 0.0
        while stack:
            v = stack.pop()
            for i in consumers[v]:
                if i in seen:
                    continue
                seen.add(i)
                if T.kind(ops[i]) == "Bootstrap":
                    continue
                dops += cost(ops[i], 0) - cost(ops[i], 2)
                stack.extend(ops[i]["outs"])
        org = next(iter((o.get("origin") or {}).values()), None)
        site = next((s for s in sites_all if s["block"] == b and s["var"] == org), None)
        dbts = 0.0
        if site and site["route"] != "1":
            s2 = dict(site, drop=(site["drop"] or 0) + 1)
            dbts = ms(T.site_units(site, sd, ss)) - ms(T.site_units(s2, sd, ss))
        rows.append((b, site["var"] if site else name_of[out], (site["route"] + ":" + site["kind"][:5]) if site else "?", site["drop"] if site else None,
                     dbts, dops))
tot = sum(r[4] + r[5] for r in rows)
print(f"{len(rows)} refreshes; +1 drop on all (upper bound, ignores feasibility): {tot:.1f} ms "
      f"(refreshes {sum(r[4] for r in rows):.1f}, downstream ops {sum(r[5] for r in rows):.1f})")
by = collections.defaultdict(list)
for r in rows:
    by[r[2]].append(r[4] + r[5])
for k, v in by.items():
    v.sort()
    print(f"  route {k:5s} n {len(v):3d}  value per +1 drop: median {v[len(v) // 2]:.2f} ms, max {v[-1]:.2f}")
print("  top sites:", [(r[0], r[1], r[2], r[3], round(r[4] + r[5], 2)) for r in sorted(rows, key=lambda r: -(r[4] + r[5]))[:12]])

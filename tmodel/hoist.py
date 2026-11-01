"""Rotations (Rotate/Conjugate) that share their input ciphertext with another rotation in the same block: candidates
for a shared ModUp (hoisting). Also the Realize ops. usage: python -I hoist.py PROGS PLAN"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokplan as T, units as U  # noqa: E402
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")))
ms = lambda u: C["tau"] * u[0] + C["kappa"] * u[1] + C["eps"] * u[2]  # noqa: E731
d = sys.argv[1] + "/" + sys.argv[2]
save = 0.0; groups = collections.Counter(); real = collections.Counter(); realize_ms = 0.0
for b in range(14):
    p = json.load(open(f"{d}/block_{b}.json")); V = p["vars"]
    lv = lambda i: V[i]["meta"]["level"]  # noqa: E731
    ops = []
    def flat(o):
        if T.kind(o) == "Batched":
            for s in o["kind"]["Batched"]:
                flat(s)
        else:
            ops.append(o)
    for o in p["ops"]:
        flat(o)
    byin = collections.defaultdict(list)
    for o in ops:
        if T.kind(o) in ("Rotate", "Conjugate", "RotateSum") and not any(x for x in o["pre"]):
            byin[o["ins"][0]].append(o)
        if "Realize" in (o.get("origin") or {}):
            acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
            T.op_units(o, lv, acc); realize_ms += sum(ms(acc[c]) for c in ("ks", "rescale", "ew"))
            real[T.kind(o) + str([next(iter(x)) if isinstance(x, dict) else x for x in o["pre"]])] += 1
    for i, g in byin.items():
        if len(g) < 2:
            continue
        n = T.NQ - lv(i)
        groups[len(g)] += 1
        # each extra rotation in the group skips its ModUp
        save += (len(g) - 1) * ms(U.modup(n, T.CH))
print(f"rotation groups sharing an input (size: count): {dict(sorted(groups.items()))}")
print(f"ModUp saved by hoisting them: {save:.1f} ms/token")
print(f"Realize ops: {realize_ms:.1f} ms/token, kinds {dict(real.most_common(8))}")

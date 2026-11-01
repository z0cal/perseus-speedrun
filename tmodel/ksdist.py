"""Where the token's key-switch work sits by limb count (record plan), and the op time if every op ran 2 limbs
higher/lower (the level effect). usage: python -I ksdist.py PROGS_DIR PLAN"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokplan as T, units as U  # noqa: E402
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")))
ms = lambda u: C["tau"] * u[0] + C["kappa"] * u[1] + C["eps"] * u[2]  # noqa: E731
d = sys.argv[1] + "/" + sys.argv[2]
for shift in (0, -2, +2):
    byn = collections.defaultdict(float); kinds = collections.defaultdict(float)
    tot = 0.0
    for b in range(14):
        p = json.load(open(f"{d}/block_{b}.json"))
        V = p["vars"]
        lv = lambda i: V[i]["meta"]["level"] + shift  # noqa: E731
        for o in p["ops"]:
            acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
            T.op_units(o, lv, acc)
            k = T.kind(o)
            outs = [lv(i) for i in o["outs"]] or [lv(i) for i in o["ins"]] or [0]
            n = T.NQ - max(outs)
            t = sum(ms(acc[c]) for c in ("ks", "rescale", "ew"))
            tot += t
            byn[n] += ms(acc["ks"]); kinds[k] += t
    print(f"shift {shift:+d} (levels; + = fewer limbs): ops {tot:.1f} ms")
    if shift == 0:
        print("  key-switch ms by limb count n:", {n: round(v, 1) for n, v in sorted(byn.items()) if v > 0.5})
        print("  ms by op kind:", {k: round(v, 1) for k, v in sorted(kinds.items(), key=lambda x: -x[1])})

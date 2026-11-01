"""Op census of a compiled Decode-8 token (14 stage programs): per kind, count and sum of limbs (51 - level on n32a4)."""
import collections, json, sys
NQ = 51
def kind(o):
    return o["kind"] if isinstance(o["kind"], str) else next(iter(o["kind"]))
def census(dirn):
    C = collections.defaultdict(lambda: [0, 0])  # kind -> [count, limb-sum]
    bts = []
    for b in range(14):
        d = json.load(open(f"{dirn}/block_{b}.json"))
        V = d["vars"]
        lv = lambda i: V[i]["meta"]["level"]
        def visit(o):
            k = kind(o)
            if k == "Batched":
                for s in o["kind"]["Batched"]:
                    visit(s)
                return
            ins = [lv(i) for i in o["ins"]]
            outs = [lv(i) for i in o["outs"]]
            for p in o["pre"]:
                pk = next(iter(p)) if isinstance(p, dict) else p
                if pk == "Rescale":
                    # the operand is rescaled before the op: count a rescale at the op's input level - 2 (pre-rescale)
                    n = NQ - (min(outs) if outs else min(ins)) + 2
                    C["pre_rescale"][0] += 1; C["pre_rescale"][1] += n
            n = NQ - (max(outs) if outs else max(ins))  # limbs the op works on (after its pre-rescales)
            if k == "Bootstrap":
                return
            if k == "RotateSum":
                C["rot_in_sum"][0] += len(o["kind"]["RotateSum"]); C["rot_in_sum"][1] += n * len(o["kind"]["RotateSum"])
            C[k][0] += 1; C[k][1] += n
        for o in d["ops"]:
            visit(o)
        for s in d["bts_sites"]:
            bts.append(s)
    return dict(C), bts
if __name__ == "__main__":
    res = {}
    for name in sys.argv[2:]:
        C, bts = census(f"{sys.argv[1]}/{name}")
        res[name] = C
    keys = sorted(set(k for C in res.values() for k in C))
    print(f"{'kind':14s}" + "".join(f"{n:>22s}" for n in res))
    for k in keys:
        print(f"{k:14s}" + "".join(f"{res[n].get(k,[0,0])[0]:>10d}{res[n].get(k,[0,0])[1]:>12d}" for n in res))

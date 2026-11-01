"""Token-level work units of a compiled Decode-8 plan (the 14 stage programs saved by saveprogs.py) on n32a4:
ops (digit-aware key switches, hoisted rotate-sums, rescales, elementwise) and refresh sites (route, real, raise drop,
EvalMod series). usage: python -I tokplan.py PROGS_DIR PLAN... (prints the unit totals per plan)."""
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import units as U  # noqa: E402

CH = "n32a4"
NQ = U.CHAINS[CH]["q"]
H = os.path.dirname(os.path.abspath(__file__))


def kind(o):
    return o["kind"] if isinstance(o["kind"], str) else next(iter(o["kind"]))


EW = {"Add": 2, "Sub": 2, "MultConst": 2, "AddConst": 1, "MultI": 2, "Negate": 2, "AddPt": 1, "MultPt": 3}


def op_units(o, lv, acc):
    """Add op o's units to acc (dict of feature lists per class)."""
    k = kind(o)
    if k == "Batched":
        for s in o["kind"]["Batched"]:
            op_units(s, lv, acc)
        return
    if k == "Bootstrap":
        return
    outs = [lv(i) for i in o["outs"]]
    ins = [lv(i) for i in o["ins"]]
    n = NQ - (max(outs) if outs else max(ins))
    for p in o["pre"]:
        pk = next(iter(p)) if isinstance(p, dict) else p
        if pk == "Rescale":
            U.add(acc["rescale"], U.rescale(n + 2))
            acc["launch"][0] += 1
    if k in ("Rotate", "Conjugate"):
        U.add(acc["ks"], U.ks(n, CH))
        acc["launch"][0] += 3
    elif k == "RotateSum":
        r = len(o["kind"]["RotateSum"])
        U.add(acc["ks"], U.modup(n, CH))
        for _ in range(r):
            U.add(acc["ks"], U.ks_hoisted_extra(n, CH))
        U.add(acc["ks"], U.moddown(n, CH))
        acc["launch"][0] += 3
    elif k in ("Mult", "Square"):
        U.add(acc["ew"], [0, 0, 4 * n])
        U.add(acc["ks"], U.ks(n, CH))
        acc["launch"][0] += 3
    elif k == "Rescale":
        U.add(acc["rescale"], U.rescale(n + 2))
        acc["launch"][0] += 1
    elif k in EW:
        U.add(acc["ew"], [0, 0, EW[k] * n])
        acc["launch"][0] += 1
    else:
        raise ValueError(k)


def plan_units(dirn):
    acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
    sites = []
    for b in range(14):
        d = json.load(open(f"{dirn}/block_{b}.json"))
        V = d["vars"]
        lv = lambda i: V[i]["meta"]["level"]  # noqa: E731
        for o in d["ops"]:
            op_units(o, lv, acc)
        for s in d["bts_sites"]:
            r = s["route"]
            route = "stcf" if (r["slots"] == 0 and r["stc_first"]) else ("dense" if r["slots"] == 0 else str(r["slots"]))
            sites.append(dict(block=b, var=s["var"], kind=s["kind"], route=route, real=r["real"], drop=r["raise_drop"],
                              cf=r["cf"], lin=s["input"]["level"], lout=s["landing"]["level"]))
    cfg = json.loads(d["cfg_stamp"])["config"]
    return acc, sites, cfg


# ── refresh sites ──
def em_list(top, series, k24):
    em = U.load_em(f"{H}/em/{CH}_t{top}.jsonl")
    name = {("shipped", False): "Shipped_K16", ("shipped", True): "Shipped_K24", ("d27", False): "D27_K16",
            ("d27", True): "D27_K24"}.get((series, k24))
    if name is None:  # a candidate file name (K16 fit used for both K, as a cost proxy)
        name = next(k for k in em if k.endswith(series))
    return em[name]["oplist"]


def site_units(s, series_dense, series_sparse, spru_ms_units=None):
    """Units of one refresh on n32a4 (K = 1). stcf: Split (StC on the input side) + raise + 4 CtS + EvalMod (1 chain
    real, 2 complex). SparseB-512: raise + fold + 1 CtS stage + accumulate + 1 EvalMod chain (K24 table) + 1 StC stage
    + accumulate. SPRU: returned as None (priced separately). A raise drop d lowers the raise top by 2d limbs."""
    d = s["drop"] or 0
    u = U.z()
    if s["route"] in ("stcf", "dense"):
        top_n = NQ - 2 * d  # raise to 51 limbs, minus the drop
        U.add(u, U.raise_(top_n))
        U.add(u, U.lt_stage(top_n, 16, 2, 31, CH, giant_identity=True))
        U.add(u, U.lt_stage(top_n - 2, 16, 2, 31, CH))
        U.add(u, U.lt_stage(top_n - 4, 16, 2, 31, CH))
        U.add(u, U.lt_stage(top_n - 6, 8, 2, 15, CH))
        top = 42 - 2 * d
        em = U.evalmod(em_list(top, series_dense, False), CH)
        U.add(u, em, 1 if s["real"] else 2)
        if s["route"] == "stcf":
            # StC on the input side: 3 stages on 7..3 single-prime limbs (pc-bench "Split" 1.41 ms on n32: priced as
            # 3 StC stages at n = 7)
            for _ in range(3):
                U.add(u, U.lt_stage(7, 16, 4, 63, CH))
        else:
            for n in (26, 24, 22):
                U.add(u, U.lt_stage(n, 16, 4, 63, CH))
        return u
    if s["route"] == "512":
        top_n = NQ - 2 * d
        U.add(u, U.raise_(top_n))
        U.add(u, U.lt_stage(top_n, 16, 2, 32, CH))
        U.add(u, U.modup(top_n - 2, CH))  # accumulate: 6 rotate-adds, two hoisted rounds of 3
        for _ in range(6):
            U.add(u, U.ks_hoisted_extra(top_n - 2, CH))
        U.add(u, U.moddown(top_n - 2, CH))
        U.add(u, U.moddown(top_n - 2, CH))
        top = 48 - 2 * d
        U.add(u, U.evalmod(em_list(top, series_sparse, True), CH))
        n_stc = NQ - s["lout"] + 2
        U.add(u, U.lt_stage(n_stc, 16, 2, 32, CH))
        U.add(u, U.modup(n_stc, CH))
        for _ in range(6):
            U.add(u, U.ks_hoisted_extra(n_stc, CH))
        U.add(u, U.moddown(n_stc, CH))
        return u
    return None


def series_of(cfg):
    em = cfg["bts"].get("evalmod") or "shipped"
    return {"shipped": ("shipped", "shipped"), "d27": ("d27", "d27"), "d27sb": ("shipped", "d27")}[em]


if __name__ == "__main__":
    for name in sys.argv[2:]:
        acc, sites, cfg = plan_units(f"{sys.argv[1]}/{name}")
        sd, ss = series_of(cfg)
        print(name, "evalmod", sd, ss, "sites", collections.Counter(s["route"] for s in sites))
        for c, v in acc.items():
            print(f"   ops {c:8s} ntt {v[0]:10.0f} conv {v[1]:10.0f} ew {v[2]:10.0f}")

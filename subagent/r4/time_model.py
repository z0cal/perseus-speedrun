"""r4: refresh mix (all 14 stages) and a bootstrap-time model for Decode-8 plans on record 4's chain.
bts_sites tuples: (name, kind, route, cf, real, raise_drop, stc_first). Costs: plan/data/route_table_n32.json bts_ms
(record 3's chain) x record 4's README ratios on n32a4 (StC-first 0.84, SparseB 0.82, SPRU 0.81; the shipped-order dense
taken at 0.84 too). The d27 EvalMod's extra cost on dense routes: +EM_SHARE x 20 % (auditoria R3: +20 % limb-products
in the dense EvalMod; SparseB's K24 table costs the same: 98 vs 97 ops).
    python time_model.py GRAPH PLAN [GRAPH PLAN ...]     (from work/perseus-r4, PYTHONPATH=python)"""
import collections, json, sys
from perseus import _pc

B = [f"block_{b}" for b in range(14)]
T = json.load(open("python/perseus/plan/data/route_table_n32.json"))["bts_ms"]
LEAN = {"stcf": 0.84 * T["stcf"], "dense": 0.84 * T["dense"], "512": 0.82 * T["512"], "1": 0.81 * T["1"]}


def key(x):
    if x[2] == 0:
        return "stcf" if x[6] else "dense"
    return str(x[2])


rows = {}
args = sys.argv[1:]
for g, p in zip(args[::2], args[1::2]):
    ps = _pc.Program.compile(g, p, blocks=B, keep_going=True)
    errs = sum(len(ps[b].check[1]) for b in B)
    sites = [x for b in B for x in ps[b].bts_sites]
    mix = collections.Counter(key(x) for x in sites)
    rd = collections.Counter((key(x), x[5]) for x in sites if x[5])
    ms = sum(LEAN[k] * n for k, n in mix.items())
    dense_n = mix["stcf"] + mix["dense"]
    rows[p] = (len(sites), dict(mix), ms, dense_n)
    print(f"{p}: {len(sites)} refreshes/token, errors {errs}, mix {dict(sorted(mix.items()))}, raise drops {dict(sorted(rd.items()))}")
    print(f"   bootstrap ms (route_table x lean ratios) {ms:.1f}; dense-EvalMod refreshes {dense_n}")
print("lean per-route ms:", {k: round(v, 2) for k, v in LEAN.items()})

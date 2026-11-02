"""Refresh cost of the record-4 plan under EvalMod variants, from its placement files (no compiled programs needed):
route (sparse_slots: 512 = SparseB, 1 = SPRU, else dense = StC-first on this runtime), real (real_route), raise drop
(raise_drop, else the runtime base 512:1), priced with tokplan.site_units and the calibrated units (calib.json).
The landings do not change in these scenarios (each extra level is spent on one more raise drop), so the plan's ops
are the same and only the refreshes move. SPRU is fixed (8 ms, not modelled), as in predict.py.
The even series' op lists (em/even_n32a4_t{top}.jsonl) come from emcost_even.rs (pc-shadow example on upstream
d97a6f7: the census of emcost_main.rs plus the even prestep; its shipped/d27 lists equal em/n32a4_t*.jsonl exactly).
usage: python -I even_sites.py PLACEMENTS_DIR   (the record-4 plan: bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm)"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P  # noqa: E402
import tokplan as T  # noqa: E402
import units as U  # noqa: E402

_load = U.load_em


def _load_with_even(path):
    """em/n32a4_t{top}.jsonl plus the even series of em/even_n32a4_t{top}.jsonl."""
    d = _load(path)
    ev = path.replace("/n32a4_t", "/even_n32a4_t")
    if os.path.exists(ev):
        d.update(_load(ev))
    return d


U.load_em = _load_with_even
pl = sys.argv[1]
sites = []
for b in range(14):
    p = json.load(open(f"{pl}/block_{b}_placement.json"))
    ss, rd, real = p.get("sparse_slots", {}), p.get("raise_drop", {}), set(p.get("real_route", []))
    vs = [x["target_var"] for x in p["placements"]] + list(p.get("hint_fire", []))
    for v in vs:
        slots = ss.get(v, 0)
        route = {512: "512", 1: "1"}.get(slots, "stcf")
        sites.append(dict(block=b, var=v, route=route, real=v in real, drop=rd.get(v, 1 if slots == 512 else 0), lout=26))
    # deliberate refreshes are not in the placement lists: the K/V push in blocks 0-11 (dense StC-first, real) and
    # 6 in block 13 (SPRU; its 6 hints are the complex CutMax pairs). This reproduces the tmodel report's mix
    # (36 StC-first real, 6 complex, 40 SparseB-512, 32 SPRU).
    for _ in range(p["summary"]["num_deliberate_bootstrap_nodes"]):
        sites.append(dict(block=b, var="delib", route="stcf" if b < 13 else "1", real=True, drop=0, lout=26))
    assert len(vs) + p["summary"]["num_deliberate_bootstrap_nodes"] == p["summary"]["total_bootstraps"], b
n = collections.Counter(s["route"] + ("r" if s["real"] else "") for s in sites)
print("sítios:", dict(n), "total", len(sites))


def total(dense, sparse, drop_add=None):
    """ms of all refreshes; drop_add = {route: extra raise drop}."""
    by = collections.defaultdict(float)
    for s in sites:
        if s["route"] == "1":
            by["1"] += P.SPRU_MS
            continue
        site = dict(s)
        site["drop"] = (site["drop"] or 0) + (drop_add or {}).get(s["route"], 0)
        by[s["route"] + ("r" if s["real"] else "")] += P.ms(T.site_units(site, dense, sparse))
    return sum(by.values()), dict(by)


base, bb = total("shipped", "shipped")
print(f"recorde: refreshes {base:.1f} ms  " + "  ".join(f"{k} {v:.1f}/{n[k]} ({v / n[k]:.2f})" for k, v in sorted(bb.items())))
rows = [
    ("A: SparseB d27 K24 + 1 drop (sbd27d1, medido x1,007)", "shipped", "d27", {"512": 1}),
    ("A': SparseB par z13 K24 + 1 drop", "shipped", "Even_z13_K24", {"512": 1}),
    ("densas d27 + 1 drop (pouso do recorde)", "d27", "shipped", {"stcf": 1}),
    ("E: densas par z11 + 1 drop (pouso do recorde)", "Even_z11_K16", "shipped", {"stcf": 1}),
    ("densas par z13 + 1 drop", "Even_z13_K16", "shipped", {"stcf": 1}),
    ("E + A': densas par z11 + SparseB par z13, +1 drop", "Even_z11_K16", "Even_z13_K24", {"stcf": 1, "512": 1}),
    ("E + A: densas par z11 + SparseB d27, +1 drop", "Even_z11_K16", "d27", {"stcf": 1, "512": 1}),
    ("(só a tabela) densas par z11 sem drop: pouso sobe, plano muda", "Even_z11_K16", "shipped", None),
]
for name, d, s, add in rows:
    t, by = total(d, s, add)
    print(f"{name:62s} {t - base:+7.1f} ms  ({100 * (t - base) / P.MEASURED['record']:+.2f} % de 1690)  "
          + "  ".join(f"{k} {by[k] - bb[k]:+.1f}" for k in sorted(by) if k != "1"))

"""Token time of compiled plans from the calibrated units (calib.json): refreshes + ops, against the GPU A/Bs.
usage: python -I predict.py PROGS_DIR [plan ...]   (record first: deltas are against it)."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokplan as T  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(f"{H}/calib.json"))
# per-ct constants not in the stage units: SPRU (no measurement: route_table ratio to StC-first x n32a4 factor) and
# a per-launch cost for the token's ops (CUDA-graph replay); both fitted below against the record's 1690 ms.
SPRU_MS = float(os.environ.get("SPRU_MS", "8.0"))
LAUNCH_US = float(os.environ.get("LAUNCH_US", "0"))
MEASURED = {"record": 1690.0, "em9": 1690.0 + 76.6, "d27sb": 1690.0 + 5.9}


def ms(u):
    return C["tau"] * u[0] + C["kappa"] * u[1] + C["eps"] * u[2]


def plan_ms(dirn, override=None):
    acc, sites, cfg = T.plan_units(dirn)
    sd, ss = T.series_of(cfg)
    if override:
        sd, ss = override.get("dense", sd), override.get("sparse", ss)
    ops = sum(ms(acc[c]) for c in ("ks", "rescale", "ew")) + LAUNCH_US * 1e-3 * acc["launch"][0]
    by = collections.defaultdict(float)
    for s in sites:
        if s["route"] == "1":
            by["1"] += SPRU_MS
            continue
        site = dict(s)
        if override and "drop_add" in override and s["route"] in override.get("drop_routes", ("stcf", "512")):
            site["drop"] = (site["drop"] or 0) + override["drop_add"]
        by[s["route"] + ("r" if s["real"] else "")] += ms(T.site_units(site, sd, ss))
    return ops, dict(by), sites


if __name__ == "__main__":
    base = None
    for name in sys.argv[2:]:
        ops, by, sites = plan_ms(f"{sys.argv[1]}/{name}")
        tot = ops + sum(by.values())
        n = collections.Counter(s["route"] + ("r" if s["real"] else "") for s in sites)
        base = base if base is not None else tot
        meas = MEASURED.get(name)
        print(f"{name:8s} total {tot:7.1f} ms (delta {tot - base:+6.1f}; measured delta "
              f"{(meas - MEASURED['record']) if meas else float('nan'):+6.1f})  ops {ops:6.1f}  "
              + "  ".join(f"{k} {v:6.1f}/{n[k]} ({v / n[k]:.2f})" for k, v in sorted(by.items())))

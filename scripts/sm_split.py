"""Softmax level split per layer, with the record's round-0 band rule (v2, 2026-10-28; v1 = sm_split_v1.py).

With l1 + l2 fixed (the exp series' input scale 2^-(l1+l2) unchanged), choose l1 squarings vs l2 normalise-and-square
rounds. Round 0's band for each (layer, l2) is the record's rule (scripts/utils/fit_gpt2_decode16.py::fit_softmax):
the calibration windows' divisors d = mean_k exp((S_k - mid) / 2^l2) (the padded heads' d = exp(-mid / 2^l2) included)
widened x SAFE at both ends, and its top at least 1.05 x the divisor of a query whose scores all sit at the range top,
exp((clip_hi - mid) / 2^l2): every query inside the score range converges, whatever the data. That structural top is
what makes l2 = 0 expensive where the range is wide (it does not shrink with the data). The counts are
impl.smcheb.design's (kc 1..16, last target TGT). Levels: l1 - r + sum(counts) + l2, r = min(1, l1) the squaring
smrel=0 (subagent/search2.py) removes afterwards, so the split is chosen for the config it ends in.

Calibration: owt_windows "calib" (document starts + random offsets of pool [0, 1M), >= 16384 windows, the audit's
stress windows out); never the dev prompts.

    python -I sm_split.py [SAFE] [TGT]          (defaults 3, 1e-2)
Writes work/sm_split_rc{SAFE}_t{TGT}.json; prints per layer the record's split and the best."""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import owt_windows as ow  # noqa: E402

W = os.path.expanduser("~/pesquisa/perseus-speedrun/work")
sys.path.insert(0, f"{W}/perseus-rec/python")
from perseus.impl import smcheb  # noqa: E402

BASE = f"{W}/perseus-rec/python/perseus/configs/model/approximation/gpt2_base_n32_kc16/configs.json"
SAFE = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
TGT = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
N_WIN = 16384
L2S = range(0, 6)
KCS = tuple(k for k in smcheb.KC_DESIGN if k <= 16) + (16,)
STATS = f"{W}/round0_d_calib.json"


def mids():
    c = json.load(open(BASE))["softmax"]
    return [((s := c[f"transformer.h.{l}.attn"])["clip_hi"] + s["clip_lo"]) / 2 for l in range(12)], \
           [c[f"transformer.h.{l}.attn"]["clip_hi"] for l in range(12)]


def calib_stats():
    """{l2: [[dmin, dmax] per layer]} over the calibration windows (cached)."""
    if os.path.exists(STATS):
        d = json.load(open(STATS))
        return {int(k): v for k, v in d["by_l2"].items()}, d["n_windows"]
    docs, rnd = ow.offsets("calib", N_WIN)
    offs = (docs + rnd)[:N_WIN]
    mid, _ = mids()
    st = {l2: [[math.inf, 0.0] for _ in range(12)] for l2 in L2S}
    t0 = time.time()
    for i in range(0, len(offs), 64):
        Ss = ow.scores(ow.windows(offs[i:i + 64]))
        for l in range(12):
            for l2 in L2S:
                d = ow.divisors(Ss[l], mid[l], l2)
                r = st[l2][l]
                r[0], r[1] = min(r[0], float(d.min())), max(r[1], float(d.max()))
        if i % 2048 == 0:
            print(f"  calib {i}/{len(offs)} {time.time() - t0:.0f} s", flush=True)
    for l2 in L2S:              # the padded heads' scores are 0
        for l in range(12):
            p = math.exp(-mid[l] / 2 ** l2)
            st[l2][l] = [min(st[l2][l][0], p), max(st[l2][l][1], p)]
    json.dump({"n_windows": len(offs), "n_docstarts": len(docs), "by_l2": st}, open(STATS, "w"), indent=1)
    return st, len(offs)


def band(st, l, l2, mid, top):
    lo, hi = st[l2][l]
    return lo / SAFE, max(hi * SAFE, 1.05 * math.exp((top - mid) / 2 ** l2))


def levels(lo, hi, l1, l2):
    a, b = smcheb.cheb_seed(lo, hi)
    ks, tg = smcheb.design(a, b, l2, TGT, KCS)
    return l1 - min(1, l1) + sum(ks) + l2, ks, tg


def main():
    c = json.load(open(BASE))
    st, n = calib_stats()
    mid, top = mids()
    tot_rec = tot_best = 0
    out = {}
    for l in range(12):
        s = c["softmax"][f"transformer.h.{l}.attn"]
        l1, l2 = int(round(math.log2(s["delta1"]))), int(round(math.log2(s["delta2"])))
        rec = l1 - min(1, l1) + sum(s["recip_iters_cheb"]) + l2
        best = None
        for nl2 in range(0, l1 + l2 + 1):
            if nl2 not in st:
                continue
            lo, hi = band(st, l, nl2, mid[l], top[l])
            try:
                lv, ks, tg = levels(lo, hi, l1 + l2 - nl2, nl2)
            except ValueError:                 # a band too wide for any seed / count
                continue
            if best is None or lv < best[0]:
                best = (lv, l1 + l2 - nl2, nl2, ks, tg, lo, hi)
        tot_rec += rec
        tot_best += best[0]
        out[l] = best
        print(f"L{l:<2} record l1={l1} l2={l2} iters {s['recip_iters_cheb']} -> {rec:2d} | best l1={best[1]} l2={best[2]} "
              f"iters {best[3]} -> {best[0]:2d}  band [{best[5]:.4g}, {best[6]:.4g}] (data [{st[best[2]][l][0]:.4g}, "
              f"{st[best[2]][l][1]:.4g}])")
    print(f"softmax levels per token (smrel counted): record {tot_rec}, best (x{SAFE:g} + structural top, last target "
          f"{TGT:g}, {n} calibration windows) {tot_best}")
    json.dump({l: {"l1": b[1], "l2": b[2], "ks": b[3], "tg": b[4], "lo": b[5], "hi": b[6]} for l, b in out.items()},
              open(f"{W}/sm_split_rc{SAFE:g}_t{TGT:g}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

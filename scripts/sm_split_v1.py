"""Softmax level split per layer: with l1 + l2 fixed (the exp polynomial's input scale 2^-(l1+l2) unchanged), choose
l1 squarings vs l2 normalise-and-square rounds. Levels counted as l1 + sum(recip_iters_cheb) + l2 (one square per
refine round, the round-0 seed product inside the count). Round 0's band for each l2 comes from the calibration
windows (search.round0_ranges' windows), widened by MARGIN on both ends and capped by the record's top.

    python -I sm_split.py [MARGIN] [TGT_LAST] [TOP_MARGIN]
Prints per layer the record's split and levels, and the best split with its levels."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kl_eval  # noqa: E402
from perseus.examples.gpt2 import ref  # noqa: E402
from perseus.impl import config as pconfig, smcheb  # noqa: E402

W = os.path.expanduser("~/pesquisa/perseus-speedrun/work")
BASE = f"{W}/perseus-rec/python/perseus/configs/model/approximation/gpt2_base_n32_kc16/configs.json"
M = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
TGT = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
MHI = float(sys.argv[3]) if len(sys.argv) > 3 else M   # top margin (default: MARGIN)
KCS = tuple(k for k in smcheb.KC_DESIGN if k <= 16) + (16,)


def ranges_by_l2():
    path = f"{W}/round0_ranges_l2.json"
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    CF = pconfig.load_configs(BASE)
    SM = [CF.softmax[f"transformer.h.{l}.attn"] for l in range(12)]
    rng = {l2: [[math.inf, 0.0] for _ in range(12)] for l2 in range(6)}
    orig = ref.softmax_ref

    def tap(S, cfg, kc):
        l = SM.index(cfg)
        for l2 in rng:
            d = np.exp((S - (cfg.clip_hi + cfg.clip_lo) / 2) / 2 ** l2).sum(-1) / kc
            r = rng[l2][l]
            rng[l2][l] = [min(r[0], float(d.min())), max(r[1], float(d.max()))]
        return orig(S, cfg, kc)
    ref.softmax_ref = tap
    pool = np.load(kl_eval.POOL, mmap_mode="r")
    for i in range(256):
        o = 1_000_000 + 3889 * i
        kl_eval.forward_approx(np.array(pool[o:o + 16]), CF)
    ref.softmax_ref = orig
    json.dump(rng, open(path, "w"))
    return rng


def levels(lo, hi, l1, l2):
    a, b = smcheb.cheb_seed(lo, hi)
    ks, tg = smcheb.design(a, b, l2, TGT, KCS)
    return l1 + sum(ks) + l2, ks, tg


def main():
    c = json.load(open(BASE))
    R = ranges_by_l2()
    tot_rec = tot_best = 0
    out = {}
    for l in range(12):
        s = c["softmax"][f"transformer.h.{l}.attn"]
        l1, l2 = int(round(math.log2(s["delta1"]))), int(round(math.log2(s["delta2"])))
        rec = l1 + sum(s["recip_iters_cheb"]) + l2
        best = None
        for nl2 in range(0, l1 + l2 + 1):
            if nl2 not in R:
                continue
            lo, hi = R[nl2][l][0] / M, R[nl2][l][1] * MHI
            try:
                lv, ks, tg = levels(lo, hi, l1 + l2 - nl2, nl2)
            except ValueError:                 # a band too wide for any seed / count
                continue
            if best is None or lv < best[0]:
                best = (lv, l1 + l2 - nl2, nl2, ks, tg, lo, hi)
        tot_rec += rec
        tot_best += best[0]
        out[l] = best
        print(f"L{l:<2} record l1={l1} l2={l2} iters {s['recip_iters_cheb']} -> {rec:2d} levels | best l1={best[1]} "
              f"l2={best[2]} iters {best[3]} -> {best[0]:2d}")
    print(f"softmax levels per token: record {tot_rec}, best split (margin x{M:g}, last target {TGT:g}) {tot_best}")
    json.dump({l: {"l1": b[1], "l2": b[2], "ks": b[3], "tg": b[4], "lo": b[5], "hi": b[6]} for l, b in out.items()},
              open(f"{W}/sm_split_m{M:g}{'' if MHI == M else f'h{MHI:g}'}_t{TGT:g}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

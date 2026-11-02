"""H1: exp seeds cheaper than the record's, judged by the TOTAL error of the real softmax mirror (softmax_ref,
SM_CHEB_GS=1) against the record's total error. usage: seed_search.py LAYER [LAYER ...]"""
import sys, math, json
import numpy as np
import sm2
sys.path.insert(0, "/home/user/perseus-speedrun/scripts")
from gelu_bsgs_count import bsgs

KEYS = ("maxdp", "tv", "dout", "shape", "fac")


def cost(d, k1):
    a, b = bsgs([1.0] * (d + 1))
    return a + k1, b + k1


def fmt(m):
    return " ".join(f"{k} {m[k]:.2e}" for k in KEYS) + (f" NaN {m['nan']}" if m["nan"] else "")


def main():
    for L in map(int, sys.argv[1:]):
        j, cfg = sm2.layer(L)
        k1r = cfg.log2delta1
        rec = sm2.evaluate(j, cfg, N=3000, seed=1)["ALL"]
        rc = cost(len(cfg.cheb_coeffs) - 1, k1r)
        print(f"== camada {L}: recorde d{len(cfg.cheb_coeffs)-1} k1={k1r} delta2={2**cfg.log2delta2} custo {rc}: {fmt(rec)}", flush=True)
        cands = []
        for k1 in range(0, k1r + 1):
            for d in (6, 14):
                c = cost(d, k1)
                if c[0] >= rc[0]:
                    continue
                cands.append((d, k1))
        for d, k1 in cands:
            for how in ("lp",):
                if how == "lp":
                    co, eps = sm2.fit_lp(j, cfg, d, k1)
                    if co is None:
                        print(f"   d{d} k1={k1} lp: inviavel"); continue
                else:
                    co, eps = sm2.fit_interp(j, cfg, d, k1), float("nan")
                c2 = sm2.with_seed(j, cfg, co, k1)
                m = sm2.evaluate(j, c2, N=3000, seed=1)["ALL"]
                ratio = {k: m[k] / rec[k] for k in ("maxdp", "tv", "dout")}
                print(f"   d{d:2d} k1={k1} {how:6s} custo {cost(d, k1)} eps {eps:.1e} sum|c| {np.abs(co).sum():.2f}: {fmt(m)}"
                      f"  | razao maxdp {ratio['maxdp']:.2f} tv {ratio['tv']:.2f} dout {ratio['dout']:.2f}", flush=True)


if __name__ == "__main__":
    main()

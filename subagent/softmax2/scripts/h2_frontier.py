"""H2: Goldschmidt counts (smcheb.design, kc 1..16, the record's round-0 band) against the last round's target, per
layer; and the measured factor error for the record's counts with the last count -1/+1."""
import dataclasses, sys
import numpy as np
import sm2
from perseus.impl import smcheb

for L in range(12):
    j, cfg = sm2.layer(L)
    R = cfg.log2delta2
    row = []
    for tgt in (3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6):
        ks, tg = smcheb.design(cfg.init_alpha, cfg.init_beta, R, tgt_last=tgt, kcs=tuple(range(1, 17)))
        row.append(f"{tgt:.0e}:{ks}={sum(ks)}")
    lo0, hi0 = smcheb.band_from_seed(cfg.init_alpha, cfg.init_beta)
    rec = [int(x) for x in cfg.recip_iters_cheb]
    print(f"camada {L}: banda0 [{lo0:.3g}, {hi0:.3g}] recorde {rec} alvos {list(cfg.recip_targets)} | " + "  ".join(row))
    # last-round residual for kc = 16 and actual measured factor error
    lo, hi = smcheb.refine_band(16, float(cfg.recip_targets[-2])) if R else (lo0, hi0)
    res = [f"k={k}:{smcheb.resid(lo, hi, k):.1e}" for k in range(rec[-1] - 1, rec[-1] + 2)]
    meas = []
    for dk in (-1, 0, 1):
        ks = rec[:-1] + [rec[-1] + dk]
        c2 = dataclasses.replace(cfg, recip_iters_cheb=tuple(ks))
        m = sm2.evaluate(j, c2, N=1500, seed=2, kinds=("uniform", "real", "flat"))["ALL"]
        meas.append(f"last {ks[-1]}: fac {m['fac']:.1e} maxdp {m['maxdp']:.1e} dout {m['dout']:.1e}")
    print("    residuo kc16 " + " ".join(res) + " | medido: " + " ; ".join(meas))

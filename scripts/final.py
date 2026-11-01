"""(1) the record's own round-0 band: how far inside it the dev data stays, per layer; (2) KL of the env x5 per-kc
band calibrated on half A, tested on half B. usage: as perkc.py (PROMPTS CONFIG)."""
import os
import sys

os.environ["SM_CHEB_GS"] = "1"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import perkc as P  # noqa: E402
import sweep as S  # noqa: E402  (prints its table on import)
from perseus.impl import smcheb  # noqa: E402

M, ENV = 5.0, 2
allr = {k: np.concatenate([S.A[k], S.B[k]]) for k in S.A}
print("\nrecord band headroom over all 32 prompts (band edge / data extreme; smaller = tighter):")
for l in range(P.L):
    glo, ghi = smcheb.band_from_seed(P.SM[l].init_alpha, P.SM[l].init_beta)
    dmin = min(allr[(l, kc)].min() for kc in range(1, P.T + 1))
    dmax = max(allr[(l, kc)].max() for kc in range(1, P.T + 1))
    print(f"  L{l:<2} low {dmin / glo:6.1f}x  high {ghi / dmax:6.1f}x")

P.band, P.newk = {}, []
for l in range(P.L):
    glo, ghi = smcheb.band_from_seed(P.SM[l].init_alpha, P.SM[l].init_beta)
    k0 = 0
    for kc in range(1, P.T + 1):
        near = [S.A[(l, j)] for j in range(max(1, kc - ENV), min(P.T, kc + ENV) + 1)]
        lo, hi = max(min(x.min() for x in near) / M, glo), min(max(x.max() for x in near) * M, ghi)
        P.band[(l, kc)] = (lo, hi)
        k0 = max(k0, smcheb.k_for(lo, hi, float(P.SM[l].recip_targets[0])))
    P.newk.append(k0)
print(f"\nenv x{M:g}: round-0 iterations per layer {P.newk} (record {[int(c.recip_iters_cheb[0]) for c in P.SM]})")
test = P.prompts[S.half:]
base = P.run(test, "record (baseline), half B")
orig = smcheb.rounds


def rounds_perkc(cfg, kc):
    out = orig(cfg, kc)
    l = P.SM.index(cfg)
    lo, hi = P.band[(l, kc)]
    a, b = smcheb.cheb_seed(lo, hi)
    kap, k0, mu = smcheb.step_consts(smcheb.z0_of(lo, hi), P.newk[l])
    out[0] = (a * k0, b * k0, P.newk[l], 1.0, (kap, mu))
    return out


smcheb.rounds = rounds_perkc
new = P.run(test, f"per-kc env x{M:g}, half B")
r = new / base
print(f"ratio geo-mean {np.exp(np.log(r).mean()):.3f}, max {r.max():.3f}")

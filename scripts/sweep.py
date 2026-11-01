"""Margin sweep of the per-kc round-0 band (perkc.py): round-0 levels per token (record: sum of recip_iters_cheb[0])
and test rows outside their band, calibrating on one half of the prompts and testing on the other, both ways.
'env' = per kc, the union of the calibration ranges of kc-2..kc+2. usage: as perkc.py (PROMPTS CONFIG)."""
import os
import sys

os.environ["SM_CHEB_GS"] = "1"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import perkc as P  # noqa: E402
from perseus.impl import smcheb  # noqa: E402

T, L, SM = P.T, P.L, P.SM


def ranges(prompts):
    r = {}

    def tap(l, kc, S):
        d = P.round0_d(SM[l], kc, S)
        r.setdefault((l, kc), []).append(d)
    for toks in prompts:
        P.forward_approx(np.array(toks[:T]), tap)
    return {k: np.concatenate(v) for k, v in r.items()}


half = len(P.prompts) // 2
A, B = ranges(P.prompts[:half]), ranges(P.prompts[half:])
rec = sum(int(c.recip_iters_cheb[0]) for c in SM)
print(f"record round-0 levels/token {rec}; {sum(v.size for v in A.values())} rows per half")
print(f"{'band':>4} {'margin':>6} {'levels A->B':>11} {'out A->B':>8} {'levels B->A':>11} {'out B->A':>8}  worst excess")
for env in (0, 2):
    for m in (3, 5, 10, 20, 50):
        row = []
        worst = 0.0
        for cal, tst in ((A, B), (B, A)):
            lev, out = 0, 0
            for l in range(L):
                glo, ghi = smcheb.band_from_seed(SM[l].init_alpha, SM[l].init_beta)
                tg = float(SM[l].recip_targets[0])
                k0 = 0
                for kc in range(1, T + 1):
                    near = [cal[(l, j)] for j in range(max(1, kc - env), min(T, kc + env) + 1)]
                    lo = max(min(x.min() for x in near) / m, glo)
                    hi = min(max(x.max() for x in near) * m, ghi)
                    k0 = max(k0, smcheb.k_for(lo, hi, tg))
                    d = tst[(l, kc)]
                    out += int(((d < lo) | (d > hi)).sum())
                    worst = max(worst, (d.max() / hi), (lo / d.min()))
                lev += k0
            row += [lev, out]
        print(f"{'env' if env else 'kc':>4} {m:>6} {row[0]:>11} {row[1]:>8} {row[2]:>11} {row[3]:>8}  {worst:.2f}x")

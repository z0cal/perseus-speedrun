"""Global (kc-independent) round-0 band with only the top edge tightened to M x the calibration max (the record's low
edge kept); levels per token and out-of-band test rows, both halves. usage: as perkc.py."""
import os
import sys

os.environ["SM_CHEB_GS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import perkc as P  # noqa: E402
import sweep as S  # noqa: E402
from perseus.impl import smcheb  # noqa: E402

print()
for M in (5, 10, 20):
    res = []
    for cal, tst in ((S.A, S.B), (S.B, S.A)):
        lev = out = 0
        for l in range(P.L):
            glo, ghi = smcheb.band_from_seed(P.SM[l].init_alpha, P.SM[l].init_beta)
            hi = min(ghi, M * max(cal[(l, kc)].max() for kc in range(1, P.T + 1)))
            lev += smcheb.k_for(glo, hi, float(P.SM[l].recip_targets[0]))
            out += sum(int(((tst[(l, kc)] < glo) | (tst[(l, kc)] > hi)).sum()) for kc in range(1, P.T + 1))
        res += [lev, out]
    print(f"global, top x{M}: levels {res[0]}/{res[2]} (record 88), out-of-band {res[1]}/{res[3]}")

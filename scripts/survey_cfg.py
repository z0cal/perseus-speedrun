"""Per-layer degrees and errors of an approximation config's GELU, LayerNorm and softmax.
usage: python -I survey_cfg.py CONFIG_JSON"""
import json
import math
import sys

import numpy as np
from numpy.polynomial import chebyshev as C

c = json.load(open(sys.argv[1]))
gelu = lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
t = np.cos(np.linspace(0, math.pi, 20001))
layer = lambda kv: int(kv[0].split(".")[2]) if kv[0].split(".")[1] == "h" else 99
print("GELU (deg, xmax, max err in t units):")
for k, g in sorted(c["softgelu"].items(), key=layer):
    e = np.abs(C.chebval(t, g["cheb_coeffs"]) - gelu(g["xmax"] * t) / g["xmax"]).max()
    print(f"  {k:26s} deg {len(g['cheb_coeffs']) - 1:3d} xmax {g['xmax']:6.1f} err {e:.1e}")
print("LayerNorm (deg, range ratio, rel err vs inv_out_scale/sqrt):")
for k, n in sorted(c["norm"].items(), key=layer):
    lo, hi = n["cheb_lo"], n["cheb_hi"]
    x = lo + (hi - lo) * (t + 1) / 2
    e = np.abs(C.chebval(t, n["cheb_coeffs"]) * np.sqrt(x) / n["inv_out_scale"] - 1).max()
    print(f"  {k:26s} deg {len(n['cheb_coeffs']) - 1:3d} ratio {hi / lo:6.1f} relerr {e:.1e}")
print("softmax (delta1, delta2, exp deg, recip_iters_cheb, targets):")
for k, s in sorted(c["softmax"].items(), key=layer):
    print(f"  {k:26s} d1 {s['delta1']} d2 {s['delta2']} expdeg {len(s['cheb_coeffs']) - 1} "
          f"iters {s.get('recip_iters_cheb')} tg {s.get('recip_targets')}")

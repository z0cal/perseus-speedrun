"""Scan of EVEN-in-(u-1/4) EvalMod series: depth model levels = 1 (z square) + D(dz) + r, D = 3 (dz<=5), 4 (<=13),
5 (<=27). Prints bits at eps = 2^-2, 2^-6, 2^-10 for K = 16 and K = 24 (SparseB). usage: python -I em_even_scan.py OUTDIR"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import em_even as V

out = sys.argv[1]
D = lambda d: 3 if d <= 5 else 4 if d <= 13 else 5 if d <= 27 else 6
for K in (16, 24):
    print(f"== K={K}")
    print(f"{'dz':>3} {'r':>2} {'lvls':>4} {'fit':>6} {'sum|c|':>7}   2^-2  2^-6  2^-10")
    for r in (3, 4, 5):
        for dz in (5, 7, 9, 11, 13, 15, 20):
            for fe in (0.5, 2.0 ** -6):
                try:
                    c = V.fit_even(dz, r, fe, float(K), K)
                except Exception as e:
                    continue
                bits = [-math.log2(V.err_even(c, r, 2.0 ** -x, float(K), K)) for x in (2, 6, 10)]
                lv = 1 + D(dz) + r
                print(f"{dz:3d} {r:2d} {lv:4d} {fe:6.3f} {np.abs(c).sum():7.2f}  " + " ".join(f"{b:5.1f}" for b in bits))
                json.dump(list(c), open(os.path.join(out, f"k{K}_z{dz}_r{r}_fe{'full' if fe == 0.5 else 'e6'}.json"), "w"))

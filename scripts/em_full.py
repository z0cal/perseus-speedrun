"""Whole-interval (eps-agnostic) minimax refits of EvalMod, evaluated up to eps = 2^-2. usage: python -I em_full.py OUTDIR"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import em_fit as E  # noqa: E402

out = sys.argv[1]
ks = range(2, 13)
print(f"{'config':26s} {'sum|c|':>7} " + " ".join(f"{'2^-' + str(k):>6}" for k in ks))
def row(name, c, r):
    print(f"{name:26s} {np.abs(c).sum():7.2f} " + " ".join(f"{-math.log2(E.err(c, r, 2.0 ** -k)):6.1f}" for k in ks))
row("shipped d14 r5 (20 limbs)", E.SHIPPED, 5)
for deg, r, limbs in [(13, 5, 18), (20, 4, 18), (27, 4, 18), (27, 3, 16), (59, 3, 18)]:
    c = E.fit(deg, r, 0.5)
    json.dump(list(c), open(os.path.join(out, f"full_d{deg}_r{r}.json"), "w"))
    row(f"full d{deg} r{r} ({limbs} limbs)", c, r)

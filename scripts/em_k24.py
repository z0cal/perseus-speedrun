"""SparseB's K = 24 EvalMod (degree 20 + 5 double angles) against one-level-cheaper whole-interval refits.
usage: python -I em_k24.py OUTDIR"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import em_fit as E  # noqa: E402

E.K = 24
K24 = [-5.01431878575762102e-01, -2.60978240793176830e-02, -2.75969415548868335e-01, -3.76057943042461193e-02,
       6.98674483745098551e-01, 2.06638885145157200e-02, -1.93915485831718332e-01, -3.59503488102385325e-03,
       2.34902992390395096e-02, 3.23162556614403976e-04, -1.63628724411827371e-03, -1.80046855470203313e-05,
       7.47055232583680419e-05, 6.86710430736672347e-07, -2.41801471221991641e-06, -1.91108603715104241e-08,
       5.85030407613337840e-08, 4.05831352269635484e-10, -1.09946346874119116e-09, -6.79891244700694627e-12,
       1.65253168377321969e-11]
ks = (2, 4, 6, 8, 10, 12)
print(f"{'config (K=24)':24s} {'sum|c|':>7} " + " ".join(f"{'2^-' + str(k):>6}" for k in ks))
def row(n, c, r):
    print(f"{n:24s} {np.abs(c).sum():7.2f} " + " ".join(f"{-math.log2(E.err(c, r, 2.0 ** -k)):6.1f}" for k in ks))
row("shipped d20 r5 (20)", K24, 5)
for deg, r in [(27, 4), (13, 5)]:
    c = E.fit(deg, r, 0.5)
    json.dump(list(c), open(os.path.join(sys.argv[1], f"k24_full_d{deg}_r{r}.json"), "w"))
    row(f"full d{deg} r{r} (18)", c, r)

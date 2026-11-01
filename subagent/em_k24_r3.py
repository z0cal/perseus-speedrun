"""SparseB K=24 EvalMod at 8 levels (degree 27 + 3 double angles): precision only (no file written)."""
import math, os, sys
sys.path.insert(0, os.path.expanduser("~/pesquisa/perseus-speedrun/scripts"))
import em_fit as E
E.K = 24
ks = (2, 4, 6, 8, 10, 12)
for deg, r, feps in [(27, 4, 0.5), (27, 3, 0.5), (27, 3, 2.0 ** -6), (27, 3, 2.0 ** -8)]:
    c = E.fit(deg, r, feps)
    print(f"K=24 d{deg} r{r} fit_eps={feps:g}: " + " ".join(f"2^-{k}:{-math.log2(E.err(c, r, 2.0 ** -k)):.1f}b" for k in ks))

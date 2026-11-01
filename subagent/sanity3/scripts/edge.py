import math, sys
import numpy as np
sys.path.insert(0, '.')
from emchain import chain, target, Ev
from run_prec import cfgs
print("max|err| (log2) at u = I + d, |d| <= 2^-5, for the edge integers I (fit grid covers |I| <= K-1)")
for name, c, da, K, even in cfgs[:7]:
    row = []
    for I in (-(K + 1), -K, -(K - 1), K - 1, K, K + 1):
        u = I + np.linspace(-2 ** -5, 2 ** -5, 201)
        out, _ = chain(c, da, K, u, even, Ev())
        e = np.abs(out - target(u)).max()
        row.append(f"I={I:+d}: {math.log2(e):6.1f}")
    print(f"{name:22s} " + "  ".join(row))

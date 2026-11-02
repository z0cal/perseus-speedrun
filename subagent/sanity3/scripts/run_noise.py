import json, math, sys
import numpy as np
sys.path.insert(0, '.')
from emchain import *
from run_prec import cfgs
es = 6
rng = np.random.default_rng(1)
print("max|err| bits with Gaussian noise sigma on every ct x ct product (and on the input y), eps = 2^-6 and 2^-2")
for sig_e in (None, 50, 45, 40):
    sig = 0.0 if sig_e is None else 2.0 ** -sig_e
    line = []
    for name, c, da, K, even in cfgs[:7]:
        res = []
        for e in (6, 2):
            u = grid(2.0 ** -e, K, 200)
            ev = Ev(sig, rng)
            out, _ = chain(c, da, K, u, even, ev, sigma_in=sig, rng=rng)
            res.append(-math.log2(np.abs(out - target(u)).max()))
        line.append(f"{name}: {res[0]:.1f}/{res[1]:.1f}")
    print('sigma', sig_e, ' | '.join(line))
# noise only in the products (not the input): amplification ratio
print("product-noise only (no input noise), sigma=2^-45: rms error / sigma")
for name, c, da, K, even in cfgs[:7]:
    u = grid(2.0 ** -6, K, 200)
    ev0 = Ev(); o0, _ = chain(c, da, K, u, even, ev0)
    acc = []
    for t in range(5):
        ev = Ev(2.0 ** -45, rng); o, _ = chain(c, da, K, u, even, ev)
        acc.append(o - o0)
    acc = np.array(acc)
    rms = np.sqrt((acc ** 2).mean()); mx = np.abs(acc).max()
    # input-noise only
    acc2 = []
    for t in range(5):
        o, _ = chain(c, da, K, u, even, Ev(), sigma_in=2.0 ** -45, rng=rng)
        acc2.append(o - o0)
    acc2 = np.array(acc2)
    print(f"  {name:24s} products: rms/sigma {rms / 2**-45:8.1f}  max/sigma {mx / 2**-45:8.1f} | input: rms/sigma {np.sqrt((acc2**2).mean()) / 2**-45:8.1f}")

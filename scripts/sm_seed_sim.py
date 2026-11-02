"""Monte Carlo of the record-4 softmax chain (exp seed P, delta1 squarings, delta2 normalise-and-square rounds with
EXACT division) for the record's seed vs a seed with NO squarings (k1 = 0, d = 14) fitted by the weighted band LP.
Rows: 16 keys, the reference key at u = 0 (centred layers), the others uniform / clustered in [clip_lo, clip_hi].
Reports max |p - p_exact| and KL(p_exact || p). Exact arithmetic otherwise (no CKKS noise)."""
import json, math, os, sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import sm_seed_weighted as W  # noqa
cfg = json.load(open(sys.argv[1]))
rng = np.random.default_rng(0)

def chain(P, t, k1, d2):
    z = P(t) ** (2 ** k1)
    for r in range(int(math.log2(d2)) + 1):
        y = z / z.sum(axis=1, keepdims=True)
        z = y ** 2
    return y

def fit_weighted(d, n, u, mid, H):
    e = W.best_eps(d, n, u, mid, H)
    t = (u - mid) / H; w = np.maximum(1, np.exp(u))
    lo = np.maximum(np.exp(u) - e * w, 0) ** (1 / n) * math.exp(-mid / n); hi = (np.exp(u) + e * w) ** (1 / n) * math.exp(-mid / n)
    T = np.polynomial.chebyshev.chebvander(t, d)
    r = linprog(np.zeros(d + 1), A_ub=np.vstack([T, -T]), b_ub=np.r_[hi, -lo], bounds=[(None, None)] * (d + 1), method="highs")
    return r.x, e

if __name__ == '__main__':
    for L in (0, 2, 5, 9, 11):
        s = cfg["softmax"][f"transformer.h.{L}.attn"]
        mid, H, k1, d2 = s["mid_x"], s["squeeze_bound"], int(s["n_squarings"]), int(s["delta2"])
        lo_u, hi_u = s["clip_lo"], s["clip_hi"]
        c_rec = np.array(s["cheb_coeffs"])
        c_new, e = fit_weighted(14, d2, np.linspace(lo_u, hi_u, 1500), mid, H)
        N = 20000
        u = rng.uniform(lo_u, hi_u, (N, 16))
        u[: N // 2] = np.clip(rng.normal(rng.uniform(lo_u, hi_u, (N // 2, 1)), 2.0, (N // 2, 16)), lo_u, hi_u)  # clustered rows
        if s.get("ref_keys"): u[:, 0] = 0.0
        p_ex = np.exp(u - u.max(1, keepdims=True)); p_ex /= p_ex.sum(1, keepdims=True)
        t = (u - mid) / H
        res = []
        for name, c, kk in (("recorde d8", c_rec, k1), ("k1=0 d14", c_new, 0)):
            p = chain(lambda x: np.polynomial.chebyshev.chebval(x, c), t, kk, d2)
            kl = np.sum(p_ex * (np.log(p_ex + 1e-300) - np.log(np.maximum(p, 1e-300))), 1)
            res.append(f"{name}: max|dp| {np.abs(p - p_ex).max():.1e}, KL medio {kl.mean():.1e} max {kl.max():.1e}, min p {p.min():.1e}")
        print(f"camada {L} ({'centrada' if s.get('ref_keys') else 'sem ref'}, sum|c| novo {np.abs(c_new).sum():.2f}): " + " | ".join(res))


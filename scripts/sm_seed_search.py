"""Smallest exp seed (d, k1) whose simulated softmax chain matches the record-4 max|p - p_exact| per layer (centred
layers; synthetic 16-key rows with the reference key at u = 0, exact arithmetic). usage: python -I sm_seed_search.py CONFIGS_JSON"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import sm_seed_weighted as W  # noqa
from sm_seed_sim import chain, fit_weighted  # noqa
cfg = json.load(open(sys.argv[1])); rng = np.random.default_rng(1)
for L in (0, 2, 9, 5):
    s = cfg["softmax"][f"transformer.h.{L}.attn"]
    mid, H, k1, d2 = s["mid_x"], s["squeeze_bound"], int(s["n_squarings"]), int(s["delta2"])
    lo_u, hi_u = s["clip_lo"], s["clip_hi"]
    N = 6000; u = rng.uniform(lo_u, hi_u, (N, 16))
    u[: N // 2] = np.clip(rng.normal(rng.uniform(lo_u, hi_u, (N // 2, 1)), 2.0, (N // 2, 16)), lo_u, hi_u); u[:, 0] = 0.0
    p_ex = np.exp(u - u.max(1, keepdims=True)); p_ex /= p_ex.sum(1, keepdims=True); t = (u - mid) / H
    dp = lambda c, kk: np.abs(chain(lambda x: np.polynomial.chebyshev.chebval(x, c), t, kk, d2) - p_ex).max()
    rec = dp(np.array(s["cheb_coeffs"]), k1)
    print(f"camada {L}: recorde d8 k1={k1} custo {W.cost(8, k1)} max|dp| {rec:.1e}")
    ug = np.linspace(lo_u, hi_u, 1500)
    for kk in range(0, k1 + 1):
        for d in range(3, 16):
            c, e = fit_weighted(d, 2 ** kk * d2, ug, mid, H)
            v = dp(c, kk)
            if v <= rec:
                print(f"   k1={kk}: d={d} custo {W.cost(d, kk)} max|dp| {v:.1e}"); break
        else:
            print(f"   k1={kk}: nenhum d<=15 alcanca")

"""LayerNorm inverse sqrt (LN_CHEB) per site: the shallowest chain meeting a relative-error target on the record's band
[cheb_lo, cheb_hi], among
  direct  : one relative-minimax polynomial (refit_norm_cheb.fit_site's reweighted Chebyshev fit), no Newton;
  t1      : a relative-minimax seed of degree d, then n scaled Newton steps y <- y (A_j - B_j x y^2) whose (A_j, B_j) are
            the minimax-relative ones for the band of x y^2 the previous step leaves (T1's homogeneous refiner, p = -1/2).
Depth (levels after the variance): affine 1 + ceil(log2 deg) + coefficients 1 (eval_chebyshev), minus 1 with FOLD (the
1/d and +eps folded into the interval), + 2 per Newton step (inv_sqrt_newton_d2's product order).
usage (work/perseus-r4, PYTHONPATH=python): python ln_t1.py CONFIG_JSON [TARGET] [OUT_JSON]"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "scripts/utils")
from refit_norm_cheb import fit_site  # noqa: E402
from perseus.impl.config import parse_norm  # noqa: E402

cfg_path = sys.argv[1]
target = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
out = sys.argv[3] if len(sys.argv) > 3 else None


def seed_fit(lo, hi, scale, deg):
    """Relative-minimax polynomial of degree deg for scale/sqrt(x) on [lo, hi] (fit_site's reweighting, fixed degree)."""
    t = np.cos(np.pi * (np.arange(400) + 0.5) / 400)
    xs = (t + 1) / 2 * (hi - lo) + lo
    f = scale / np.sqrt(xs)
    w = 1.0 / f
    c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w)
    for _ in range(40):
        r = np.abs(np.polynomial.chebyshev.chebval(t, c) / f - 1)
        c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w * (r / r.max() + 0.05))
    g = np.geomspace(lo, hi, 4001)
    y = np.polynomial.chebyshev.chebval((2 * g - lo - hi) / (hi - lo), c)
    e = y * np.sqrt(g) / scale            # relative: y / (scale / sqrt x)
    return c, float(e.min()), float(e.max())


def scaled_newton(emin, emax):
    """One step on u = y sqrt(x)/scale in [emin, emax]: u <- u (A - B u^2), (A, B) minimax relative (equioscillation of
    u(A - Bu^2) - 1 at emin, the interior max and emax). Returns (A, B, new range)."""
    m, M = emin, emax
    s2 = M * M + M * m + m * m              # equal ends: A = B s2
    rho = (2.0 / (3.0 * math.sqrt(3.0))) * s2 ** 1.5 / (m * M * (M + m))   # peak / end value
    E = (rho - 1.0) / (rho + 1.0)
    B = 2.0 / ((1.0 + rho) * m * M * (M + m))
    return B * s2, B, 1.0 - E, 1.0 + E


def depth(deg, nr, fold):
    return 1 + math.ceil(math.log2(deg)) + 1 - (1 if fold else 0) + 2 * nr


j = json.load(open(cfg_path))
tot = {"record": 0, "direct": 0, "best": 0}
for name, c in j["norm"].items():
    cfg = parse_norm(c)
    lo, hi, sc = c["cheb_lo"], c["cheb_hi"], cfg.inv_out_scale
    rec = depth(len(c["cheb_coeffs"]) - 1, c["cheb_nr_iters"], False)
    (_, _), ddeg, dnr, dcoef, derr = fit_site(lo, hi, sc, target, 31)
    best = (depth(ddeg, dnr, True), f"direct deg {ddeg}+NR{dnr}", None)
    for deg in range(2, 16):
        cc, emin, emax = seed_fit(lo, hi, sc, deg)
        steps = []
        for n in range(1, 3):
            A, B, emin, emax = scaled_newton(emin, emax)
            steps.append((A, B))
            if max(1 - emin, emax - 1) <= target:
                d = depth(deg, n, True)
                if d < best[0]:
                    best = (d, f"t1 seed deg {deg} + {n} scaled Newton", (cc, steps))
                break
    tot["record"] += rec
    tot["direct"] += depth(ddeg, dnr, True)
    tot["best"] += best[0]
    print(f"{name:24s} ratio {hi / lo:6.1f}  record depth {rec:2d} | direct+fold {depth(ddeg, dnr, True):2d} | "
          f"best {best[0]:2d} ({best[1]})")
print("LN levels per token after the variance:", tot)

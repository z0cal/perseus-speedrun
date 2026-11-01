"""Record-4 softmax exp under the error that matters for a reference-centred softmax: the reference key is in the row
with u = 0, so sum_j e^{u_j} >= 1 and p_i <= e^{u_i}. An error of the exp chain shows up in p_i as
|E^(u) - e^u| / max(1, e^u): ABSOLUTE for u <= 0, RELATIVE for u > 0 (first order; the delta2 rounds are exact
powers of the normalised values). E^(u) = e^mid * P(t)^n, t = (u - mid) / H, n = delta1 * delta2 (the exp's own
squarings times the rounds' squares), P a Chebyshev series of degree d on [-1, 1].
Band LP (convex in the coefficients) + bisection on eps gives the best P for each (d, k1 = log2 delta1).
usage: python -I sm_seed_weighted.py CONFIGS_JSON (numpy + scipy)"""
import json, math, os, sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gelu_bsgs_count import bsgs

def cost(d, k1):
    if d <= 3:
        return {1: (1, 0), 2: (1, 1), 3: (2, 2)}[d][0] + k1, {1: (1, 0), 2: (1, 1), 3: (2, 2)}[d][1] + k1
    # generic (all distinct, nonzero) coefficients: with all-equal ones the BSGS split cancels terms by accident and
    # undercounts by 1-2 products (degree 13/14: 5 instead of 7; subagent/softmax2/relatorio.md)
    a, b = bsgs([1.0 + 0.37 * i + 0.011 * i * i for i in range(d + 1)]); return a + k1, b + k1

def werr(E, u):
    return np.max(np.abs(E - np.exp(u)) / np.maximum(1.0, np.exp(u)))

def feasible(d, n, eps, u, mid, H):
    t = (u - mid) / H
    w = np.maximum(1.0, np.exp(u))
    lo = np.maximum(np.exp(u) - eps * w, 0.0) ** (1.0 / n) * math.exp(-mid / n)
    hi = (np.exp(u) + eps * w) ** (1.0 / n) * math.exp(-mid / n)
    T = np.polynomial.chebyshev.chebvander(t, d)
    A = np.vstack([T, -T]); b = np.r_[hi, -lo]
    r = linprog(np.zeros(d + 1), A_ub=A, b_ub=b, bounds=[(None, None)] * (d + 1), method="highs")
    return r.status == 0

def best_eps(d, n, u, mid, H):
    lo, hi = -14.0, 1.0           # log10 eps
    if not feasible(d, n, 10 ** hi, u, mid, H): return float("inf")
    for _ in range(30):
        m = 0.5 * (lo + hi)
        if feasible(d, n, 10 ** m, u, mid, H): hi = m
        else: lo = m
    return 10 ** hi

if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    for L in (0, 2, 5, 9, 11):
        s = cfg["softmax"][f"transformer.h.{L}.attn"]
        mid, H = s["mid_x"], s["squeeze_bound"]
        k1, d2 = int(s["n_squarings"]), int(s["delta2"])
        lo_u, hi_u = s["clip_lo"], s["clip_hi"]
        u = np.linspace(lo_u, hi_u, 4001)
        c = np.array(s["cheb_coeffs"], float); d0 = len(c) - 1
        # numpy convention (c0 not halved); the series is scaled by a constant (centred layers: P(1) = 1, the top of the
        # band); the constant cancels in the normalisation, so compare (P(t) / P(0))^n e^mid with e^u
        P0 = np.polynomial.chebyshev.chebval(0.0, c)
        n = 2 ** k1 * d2
        E = math.exp(mid) * (np.polynomial.chebyshev.chebval((u - mid) / H, c) / P0) ** n
        rec_w = werr(E, u)
        rec_rel = np.max(np.abs(E / np.exp(u) - 1))
        print(f"camada {L}: u em [{lo_u:.1f}, {hi_u:.1f}], mid {mid:.1f}, H {H:.1f}, recorde d={d0} k1={k1} delta2={d2}: "
              f"erro ponderado {rec_w:.1e}, relativo {rec_rel:.1e}, custo (niveis, ct*ct) {cost(d0, k1)}")
        ug = np.linspace(lo_u, hi_u, 1500)
        rows = []
        for kk in range(0, k1 + 2):
            for d in range(2, 15):
                e = best_eps(d, 2 ** kk * d2, ug, mid, H)
                if e <= rec_w * 1.0:
                    rows.append((cost(d, kk), d, kk, e)); break
        for (lv, pr), d, kk, e in sorted(rows):
            tag = " <- mais barato que o recorde" if (lv, pr) < cost(d0, k1) and pr <= cost(d0, k1)[1] else ""
            print(f"   k1={kk}: menor grau com erro ponderado <= recorde: d={d} (eps {e:.1e}), custo ({lv}, {pr}){tag}")


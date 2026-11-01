"""EvalMod with PARITY: shift u' = u - 1/4 first (AddConst, no level), then an EVEN polynomial in u'.

The pre-double-angle target s0 cos(2 pi (u - 1/4) / 2^r) = s0 cos(a u') is even in u'; the union of intervals
{I + d - 1/4} is almost symmetric about 0 (|u'| <= K - 3/4). So p(u') = sum_k c_k T_{2k}(u'/L) = sum_k c_k T_k(z),
z = 2 (u'/L)^2 - 1: ONE extra square, then a Chebyshev series of HALF the degree (a degree-13 series in z gives the
accuracy of the d27 series in u, at the nonscalar cost of the shipped degree-14 one).
usage: python -I em_even.py [OUTDIR]
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import em_fit as E  # noqa: E402

K = E.K


def pipeline_even(c, r, u, L):
    v = (u - 0.25) / L
    z = 2 * v * v - 1
    x = E.cheb(c, z)
    s = (2 * math.pi) ** (-1.0 / 2 ** r)
    for _ in range(r):
        x = 2 * x * x - s * s
        s = s * s
    return x


def err_even(c, r, eps, L, k=K):
    u = grid(eps, 400, k)
    return np.abs(pipeline_even(c, r, u, L) - np.sin(2 * math.pi * u) / (2 * math.pi)).max()


def grid(eps, n=64, k=K):
    d = np.linspace(-eps, eps, n)
    return np.concatenate([i + d for i in range(-(k - 1), k)])


def fit_even(deg_z, r, eps, L, k=K):
    u = grid(eps, 48, k)
    v = (u - 0.25) / L
    z = 2 * v * v - 1
    s0 = (2 * math.pi) ** (-1.0 / 2 ** r)
    phi = 2 * math.pi * (u - 0.25) / 2 ** r
    g = s0 * np.cos(phi)
    amp = np.ones_like(u)
    s = s0
    for j in range(r):
        amp *= np.abs(4 * s * np.cos(2 ** j * phi))
        s = s * s
    w = np.maximum(amp, 1e-3 * amp.max())
    T = np.polynomial.chebyshev.chebvander(z, deg_z)
    T[:, 0] *= 0.5
    A = np.vstack([np.hstack([w[:, None] * T, -np.ones((len(u), 1))]),
                   np.hstack([-w[:, None] * T, -np.ones((len(u), 1))])])
    b = np.concatenate([w * g, -w * g])
    res = linprog(np.r_[np.zeros(deg_z + 1), 1.0], A_ub=A, b_ub=b, bounds=[(None, None)] * (deg_z + 1) + [(0, None)],
                  method="highs")
    return res.x[:-1]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    ks = range(2, 13)
    print(f"{'config':34s} {'sum|c|':>7} " + " ".join(f"{'2^-' + str(e):>6}" for e in ks))

    def row(name, c, fn):
        print(f"{name:34s} {np.abs(c).sum():7.2f} " + " ".join(f"{-math.log2(fn(2.0 ** -e)):6.1f}" for e in ks))

    shipped = np.array(E.SHIPPED)
    row("shipped d14 r5 K16", shipped, lambda e: E.err(shipped, 5, e))
    d27 = np.array(json.load(open(os.path.join(os.path.dirname(__file__), "..", "fits", "full_d27_r4.json"))))
    row("d27 r4 K16 (em9, general)", d27, lambda e: E.err(d27, 4, e))
    for L in (K, K - 0.75 + 1e-3 + 0.0):
        for dz, r in [(13, 4), (12, 4), (11, 4), (13, 5), (6, 5), (6, 4)]:
            c = fit_even(dz, r, 0.5, L)
            row(f"EVEN z{dz} r{r} L={L:.2f}", c, lambda e, c=c, r=r, L=L: err_even(c, r, e, L))
            if out and L == K:
                json.dump(list(c), open(os.path.join(out, f"even_z{dz}_r{r}_K16.json"), "w"))


if __name__ == "__main__":
    main()

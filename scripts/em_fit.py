"""EvalMod precision: the shipped series (degree 14 + 5 double angles, K = 16) against cheaper ones.

Pipeline (FIDESlib/OpenFHE): x = sum' c_k T_k(u / K) (c0 not halved in storage), then r double angles
x <- 2 x^2 - s_j^2, s_0 = (2 pi)^(-1 / 2^r), s_{j+1} = s_j^2; the output approximates sin(2 pi u) / (2 pi) for
u = I + delta, |I| <= K - 1 integer, |delta| <= eps. E(eps) = max |out - sin(2 pi u) / (2 pi)| over that union.
Candidates are minimax (LP, Chebyshev basis) on the union of intervals at a fit eps, for configurations that cost one
level less than the shipped one on the n32 chain (pc-shadow em_depth: deg <= 13 + 5 DA, deg 14..27 + 4 DA, deg 28..59
+ 3 DA). usage: python -I em_fit.py OUTDIR
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linprog

K = 16
SHIPPED = [-5.73829476916553172e-01, 2.63718536771466207e-02, -9.15574236933522245e-01, -3.08975417541565676e-02,
           2.85601052580403802e-01, 4.83129148738224799e-03, -2.74350673185783482e-02, -3.16919293037672828e-04,
           1.31295181914509542e-03, 1.15825536926191591e-05, -3.79010152666429525e-05, -2.71035793019274615e-07,
           7.33952552563662122e-07, 4.41686895092293209e-09, -1.03169742989480305e-08]


def cheb(c, y):
    c = np.array(c, float).copy()
    c[0] /= 2
    return np.polynomial.chebyshev.chebval(y, c)


def pipeline(c, r, u):
    x = cheb(c, u / K)
    s = (2 * math.pi) ** (-1.0 / 2 ** r)
    for _ in range(r):
        x = 2 * x * x - s * s
        s = s * s
    return x


def grid(eps, n=64):
    d = np.linspace(-eps, eps, n)
    return np.concatenate([i + d for i in range(-(K - 1), K)])


def err(c, r, eps):
    u = grid(eps, 400)
    return np.abs(pipeline(c, r, u) - np.sin(2 * math.pi * u) / (2 * math.pi)).max()


def fit(deg, r, eps):
    """Minimax of the pre-double-angle target on the union of intervals; the target is s0 cos(2 pi (u - 1/4) / 2^r).
    Weighted by the first-order amplification of the r double angles, so the LP minimises the final error."""
    u = grid(eps, 48)
    y = u / K
    s0 = (2 * math.pi) ** (-1.0 / 2 ** r)
    phi = 2 * math.pi * (u - 0.25) / 2 ** r
    g = s0 * np.cos(phi)
    # d out / d x0 = prod_j 4 x_j, x_j = s_j cos(2^j phi)
    amp = np.ones_like(u)
    s = s0
    for j in range(r):
        amp *= np.abs(4 * s * np.cos(2 ** j * phi))
        s = s * s
    w = np.maximum(amp, 1e-3 * amp.max())
    T = np.polynomial.chebyshev.chebvander(y, deg)
    T[:, 0] *= 0.5                                   # storage convention: c0 not halved
    A = np.vstack([np.hstack([w[:, None] * T, -np.ones((len(u), 1))]),
                   np.hstack([-w[:, None] * T, -np.ones((len(u), 1))])])
    b = np.concatenate([w * g, -w * g])
    res = linprog(np.r_[np.zeros(deg + 1), 1.0], A_ub=A, b_ub=b, bounds=[(None, None)] * (deg + 1) + [(0, None)],
                  method="highs")
    return res.x[:-1]


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    epss = [2.0 ** -k for k in range(5, 13)]
    print("E(eps) = max |out - sin(2 pi u)/(2 pi)|, as -log2 (bits; higher is better); sum|c| = coefficient mass")
    print(f"{'config':28s} {'sum|c|':>8} " + " ".join(f"{'2^-' + str(k):>6}" for k in range(5, 13)))

    def row(name, c, r):
        print(f"{name:28s} {np.abs(c).sum():8.2f} " + " ".join(f"{-math.log2(err(c, r, e)):6.1f}" for e in epss))

    row("shipped d14 r5 (20 limbs)", SHIPPED, 5)
    print("-- one level cheaper (18 limbs):")
    best = {}
    for deg, r in [(13, 5), (27, 4), (59, 3)]:
        for ke in (6, 8, 10):
            c = fit(deg, r, 2.0 ** -ke)
            name = f"d{deg} r{r} fit@2^-{ke}"
            row(name, c, r)
            json.dump(list(c), open(os.path.join(out, f"d{deg}_r{r}_e{ke}.json"), "w"))
            best[name] = c
    print("-- two levels cheaper (16 limbs):")
    for deg, r in [(13, 4), (27, 3)]:
        for ke in (6, 8, 10):
            c = fit(deg, r, 2.0 ** -ke)
            row(f"d{deg} r{r} fit@2^-{ke}", c, r)
            json.dump(list(c), open(os.path.join(out, f"d{deg}_r{r}_e{ke}.json"), "w"))


if __name__ == "__main__":
    main()

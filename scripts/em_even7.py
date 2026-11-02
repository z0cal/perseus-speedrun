"""EvalMod PAR na MESMA profundidade do recorde: z7 + 5 ângulos duplos (10 níveis, como o grau 14 + 5 DA).

O alvo antes dos ângulos duplos, s0 cos(2 pi (u - 1/4) / 32) com u = 16 y, é par em w = y - 1/64. Com L = K = 16:

    w = y - 1/64            AddConst (sem nível)
    z = 2 w^2 - 1           1 quadrado; o x2 e o -1 vão no epílogo da relinearização (sem nível)
    p = sum c_k T_k(z)      Chebyshev de grau 7 em z (convenção OpenFHE: c0 não dividido por 2)
    5 ângulos duplos        x <- 2 x^2 - s^2, como hoje

Profundidade: 1 + PS(7) + 5 = 1 + 4 + 5 = 10 = PS(14) + 5. Com L != K seria preciso um escalar (K/L)^2 não inteiro,
que custa um nível (ou dobrá-lo no CtS); por isso L = K.

Ajuste: LP minimax ponderado pela amplificação dos ângulos duplos (`em_even7.fit`) e depois LP sequencial sobre o
erro FINAL (o pipeline dos ângulos duplos linearizado a cada rodada, com região de confiança), numa grade densa
das uniões de intervalos |I| <= K - 1, |delta| <= eps_fit.
usage: python -I em_even7.py [OUT.json]
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import em_even as V  # noqa: E402
import em_fit as E  # noqa: E402

K, R, DZ = 16, 5, 7
L = float(K)
EPS_FIT = 2.0 ** -4
N_FIT = 400


def basis(u, dz=DZ):
    v = (u - 0.25) / L
    z = 2 * v * v - 1
    T = np.polynomial.chebyshev.chebvander(z, dz)
    T[:, 0] *= 0.5
    return T


def pipeline(c, T, r=R):
    """(final value, d final / d p) over the double angles."""
    x = T @ c
    d = np.ones_like(x)
    s = (2 * math.pi) ** (-1.0 / 2 ** r)
    for _ in range(r):
        d *= 4 * x
        x = 2 * x * x - s * s
        s = s * s
    return x, d


def fit_weighted(eps, n, dz=DZ, r=R):
    """LP minimax of the pre-double-angle target, weighted by the double angles' amplification."""
    u = V.grid(eps, n, K)
    s0 = (2 * math.pi) ** (-1.0 / 2 ** r)
    phi = 2 * math.pi * (u - 0.25) / 2 ** r
    g = s0 * np.cos(phi)
    amp = np.ones_like(u)
    s = s0
    for j in range(r):
        amp *= np.abs(4 * s * np.cos(2 ** j * phi))
        s = s * s
    w = np.maximum(amp, 1e-3 * amp.max())
    T = basis(u, dz)
    A = np.vstack([np.hstack([w[:, None] * T, -np.ones((len(u), 1))]),
                   np.hstack([-w[:, None] * T, -np.ones((len(u), 1))])])
    b = np.concatenate([w * g, -w * g])
    res = linprog(np.r_[np.zeros(dz + 1), 1.0], A_ub=A, b_ub=b, bounds=[(None, None)] * (dz + 1) + [(0, None)],
                  method="highs")
    return res.x[:-1]


def refine(c, eps, n, iters=80):
    """Sequential LP on the final error, trust region grown on success and shrunk on failure."""
    u = V.grid(eps, n, K)
    tgt = np.sin(2 * math.pi * u) / (2 * math.pi)
    T = basis(u, len(c) - 1)
    e0 = np.abs(pipeline(c, T)[0] - tgt).max()
    tr = 1e-3 * np.abs(c).max()
    m = len(c)
    for _ in range(iters):
        f, d = pipeline(c, T)
        J = d[:, None] * T
        A = np.vstack([np.hstack([J, -np.ones((len(u), 1))]), np.hstack([-J, -np.ones((len(u), 1))])])
        b = np.concatenate([tgt - f, f - tgt])
        res = linprog(np.r_[np.zeros(m), 1.0], A_ub=A, b_ub=b, bounds=[(-tr, tr)] * m + [(0, None)], method="highs")
        cn = c + res.x[:-1]
        en = np.abs(pipeline(cn, T)[0] - tgt).max()
        if en < e0:
            c, e0 = cn, en
            tr *= 1.5
        else:
            tr *= 0.3
        if tr < 1e-13:
            break
    return c


def bits(c, eps, r=R):
    u = V.grid(eps, 2000, K)
    out = V.pipeline_even(c, r, u, L)
    return -math.log2(np.abs(out - np.sin(2 * math.pi * u) / (2 * math.pi)).max())


def main():
    ks = list(range(1, 13))
    print("eps = 2^-k, k =", ks)
    sh = np.array(E.SHIPPED)
    print(f"{'shipped d14 r5':24s}", [round(-math.log2(E.err(sh, 5, 2.0 ** -e)), 2) for e in ks])
    c = refine(fit_weighted(EPS_FIT, 160), EPS_FIT, N_FIT)
    b = [round(bits(c, 2.0 ** -e), 2) for e in ks]
    print(f"{'even z7 r5 (L = K)':24s}", b)
    print("coeffs", repr([float(x) for x in c]))
    if len(sys.argv) > 1:
        json.dump({"K": K, "L": L, "shift_y": 0.25 / K, "degree_z": DZ, "double_angle_its": R,
                   "eps_fit": EPS_FIT, "convention": "OpenFHE (c0 not halved)",
                   "pipeline": "w = y - 1/64; z = 2 w^2 - 1; p = sum c_k T_k(z); 5 x (x <- 2 x^2 - s^2)",
                   "bits": dict(zip([f"2^-{e}" for e in ks], b)), "coeffs": [float(x) for x in c]},
                  open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()

"""Relative minimax-ish (IRLS) fits of exp(a t) on [-1, 1]: max pointwise relative error by degree and half-width a."""
import numpy as np
from numpy.polynomial import chebyshev as C

def fit_rel(a, deg, iters=12, pad=1.0):
    n = max(16 * (deg + 1), 2000)
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n) * pad
    f = np.exp(a * t)
    w = 1.0 / f
    c = C.chebfit(t / pad, f, deg, w=w) if pad != 1 else C.chebfit(t, f, deg, w=w)
    for _ in range(iters):
        r = np.abs((C.chebval(t / pad, c) - f) / f)
        c = C.chebfit(t / pad, f, deg, w=w * np.sqrt(r / r.max() + 0.02))
    tt = np.linspace(-1, 1, 40001)
    e = np.abs(C.chebval(tt, c) / np.exp(a * pad * tt) - 1).max()
    return c, e

if __name__ == "__main__":
    for a in (1.53, 1.65, 1.75, 2.0, 2.2, 2.43):
        row = []
        for mult, deg in ((1, 8), (2, 13), (2, 14), (4, 14), (4, 27), (4, 28), (8, 28)):
            _, e = fit_rel(a * mult, deg)
            row.append(f"x{mult}/d{deg}:{e:.1e}")
        print(f"a={a:.2f}", "  ".join(row))

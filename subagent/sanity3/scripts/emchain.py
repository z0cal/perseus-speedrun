"""Independent float64 emulation of the EvalMod chain as the pc-shadow builder runs it (values only):
optional even prepass (x - 1/(4K), square, x+x, -1), PS evaluation ported from pc_core::cheb + pc_shadow::evalmod
(m=2 single level, general recursive), double angles with build_double_angle's scalars.
Optional Gaussian noise injected after every ct x ct product (relin+rescale) and on the input."""
import json, math, sys
import numpy as np

def degree(c):
    for i in range(len(c) - 1, -1, -1):
        if c[i] != 0.0:
            return i
    return 0

def not_one(v):
    return abs(v - 1.0) > 2.0 ** -44

def long_division(f, g):
    f = list(f); g = list(g)
    n = degree(f); assert n == len(f) - 1
    k = degree(g); assert k == len(g) - 1
    if n < k:
        return [0.0], f
    q = [0.0] * (n - k + 1); r = list(f); gk = g[k]
    while n > k:
        d = [0.0] * (n + 1)
        q[n - k] = 2.0 * r[-1]
        if not_one(gk): q[n - k] /= g[-1]
        nk = n - k
        if k == nk:
            d[0] = 2.0 * g[nk]
            for i in range(1, 2 * k + 1): d[i] = g[abs(nk - i)]
        elif k > nk:
            d[0] = 2.0 * g[nk]
            for i in range(1, k - nk + 1): d[i] = g[abs(nk - i)] + g[nk + i]
            for i in range(k - nk + 1, n + 1): d[i] = g[abs(i - n + k)]
        else:
            d[nk] = g[0]
            for i in range(n - 2 * k, n + 1):
                if i != nk: d[i] = g[abs(i - n + k)]
        rl = r[-1]
        if not_one(rl): d = [x * rl for x in d]
        gl = g[-1]
        if not_one(gl): d = [x / gl for x in d]
        for i in range(min(len(r), len(d))): r[i] -= d[i]
        prev = n
        if len(r) > 1:
            n = degree(r); r = r[:n + 1]
        assert n < prev or True
    if n == k:
        d = list(g); q[0] = r[-1]; gl = g[-1]
        if not_one(gl): q[0] /= gl
        rl = r[-1]
        if not_one(rl): d = [x * rl for x in d]
        if not_one(gl): d = [x / gl for x in d]
        for i in range(min(len(r), len(d))): r[i] -= d[i]
        if len(r) > 1:
            n = degree(r); r = r[:n + 1]
    q[0] *= 2.0
    return q, r

RANGE = [(2, 1), (11, 2), (13, 3), (17, 2), (55, 3), (59, 4), (76, 3), (239, 4), (247, 5)]
def ps_degrees(n):
    m = next(mm for (lim, mm) in RANGE if n <= lim)
    return n // ((1 << m) - 1) + 1, m

class Ev:
    def __init__(self, sigma=0.0, rng=None):
        self.sigma = sigma; self.rng = rng; self.nmult = 0
    def mul(self, a, b):
        self.nmult += 1
        r = a * b
        if self.sigma:
            r = r + self.rng.normal(0, self.sigma, size=np.shape(r))
        return r

def split_plan(f, k, m):
    k2m2k = k * (1 << (m - 1)) - k
    f2 = list(f[:degree(f) + 1])
    L = 2 * k2m2k + k + 1
    padded = L > len(f2)
    f2 = f2 + [0.0] * (L - len(f2))
    f2 = f2[:max(L, len(f2))]
    if padded: f2[-1] = 1.0
    tkm = [0.0] * (k2m2k + k + 1); tkm[-1] = 1.0
    dq, dr = long_division(f2, tkm)
    r2 = list(dr)
    if k2m2k - degree(dr) <= 0:
        r2[k2m2k] -= 1.0; r2 = r2[:degree(r2) + 1]
    else:
        r2 = r2 + [0.0] * (k2m2k + 1 - len(r2)); r2[-1] = -1.0
    cq, cr = long_division(r2, dq)
    s2 = list(cr) + [0.0] * max(0, k2m2k + 1 - len(cr)); s2 = s2[:k2m2k + 1]; s2[-1] = 1.0
    return dq, cq, s2

def lin(coefs, t, quirk=False):
    """c0/2 + sum c_{i+1} t[i]; quirk: FIDESlib's s2 wsum uses the first ncts T's with the filtered weights."""
    w = [(i, coefs[i + 1]) for i in range(len(coefs) - 1) if coefs[i + 1] != 0.0]
    if quirk:
        acc = sum(wi * t[j] for j, (_, wi) in enumerate(w))
    else:
        acc = sum(wi * t[i] for i, wi in w)
    return acc + coefs[0] / 2.0

def inner_ps(ev, coeffs, k, m, t, t2, lazy):
    dq, cq, s2 = split_plan(coeffs, k, m)
    dc = degree(cq)
    assert dc >= 1
    cu = t[0] * cq[1] if dc == 1 else sum(cq[i + 1] * t[i] for i in range(dc))
    cu = cu + cq[0] / 2.0
    if degree(dq) > k:
        qu = inner_ps(ev, dq, k, m - 1, t, t2, False)
    else:
        qc = (dq + [0.0] * k)[:k]
        if degree(qc) > 0:
            qu = lin(dq, t)
        else:
            qu = t[k - 1] * dq[-1] + dq[0] / 2.0
    if degree(s2) > k:
        su = inner_ps(ev, s2, k, m - 1, t, t2, False)
    else:
        sc = (s2 + [0.0] * k)[:k]
        su = lin(s2, t, quirk=True) if degree(sc) > 0 else t[k - 1] + s2[0] / 2.0
    cu = cu + t2[m - 1]
    if lazy:
        return qu, su
    return ev.mul(cu, qu) + su

def cheb_ps(ev, x, coeffs):
    n = degree(coeffs); c = list(coeffs[:n + 1])
    k, m = ps_degrees(n)
    t = [x]
    for i in range(2, k + 1):
        if i % 2 == 1:
            t.append(2 * ev.mul(t[i // 2 - 1], t[i // 2]) - x)
        else:
            t.append(2 * ev.mul(t[i // 2 - 1], t[i // 2 - 1]) - 1.0)
    t2 = [t[k - 1]]
    for i in range(1, m):
        t2.append(2 * ev.mul(t2[i - 1], t2[i - 1]) - 1.0)
    t2km1 = t2[0]
    for i in range(1, m - 1):
        t2km1 = 2 * ev.mul(t2km1, t2[i]) - t2[0]
    # m = 2 single-level shape and the general shape share the value semantics (cu qu + su - T_{k(2^m-1)})
    qu, su = inner_ps(ev, c, k, m, t, t2, True)
    cu = qu_cu = None
    dq, cq, s2 = split_plan(c, k, m)
    dc = degree(cq)
    cu = (t[0] * cq[1] if dc == 1 else sum(cq[i + 1] * t[i] for i in range(dc))) + cq[0] / 2.0 + t2[m - 1]
    return ev.mul(cu, qu) + su - 2 * ev.mul(t2km1, t2[m - 1]) + t2[0], (k, m)

def chain(c, da, K, u, even, ev, sigma_in=0.0, rng=None):
    x = u / K
    if sigma_in:
        x = x + rng.normal(0, sigma_in, size=x.shape)
    if even:
        x = x - 0.25 / K
        x = ev.mul(x, x)
        x = x + x
        x = x - 1.0
    x, km = cheb_ps(ev, x, c)
    for j in range(1, da + 1):
        x = ev.mul(x, x)
        x = x + x
        x = x - 1.0 / (2 * math.pi) ** (2.0 ** (j - da))
    return x, km

def grid(eps, K, n=400):
    d = np.linspace(-eps, eps, n)
    return np.concatenate([i + d for i in range(-(K - 1), K)])

def target(u):
    return np.sin(2 * math.pi * u) / (2 * math.pi)

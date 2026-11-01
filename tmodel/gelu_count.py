"""ct-ct products and depth of the GELU series: eval_chebyshev_bsgs (dead_ops) on y vs. the even part as a series in
u = T_2(y) = 2y^2 - 1 (T_2k(y) = T_k(u)) plus 0.5 y. usage (PYTHONPATH=python): python gelu_count.py CONFIG_JSON

2026-11-01: the series is trimmed with the config's cheb_eps (as the runtime does; the default 1e-6 drops a few more
coefficients: 205 products/token instead of 210), and the depth is carried by a mutable box so inplace_add
propagates it (with plain ints the leaf kept its first term's depth: degree 62 read as 6 levels instead of 7)."""
import json, sys
from perseus.impl import poly


class D:
    def __init__(s, d): s.d = d


class Cnt:
    cheb_bsgs = True; dead_ops = True; realize_shared = False
    def __init__(s): s.prod = 0
    def _l(s, a): return a.d if isinstance(a, D) else 0
    def square(s, a): s.prod += 1; return D(s._l(a) + 1)
    def mult(s, a, b):
        if isinstance(b, float): return D(s._l(a) + 1)
        s.prod += 1; return D(max(s._l(a), s._l(b)) + 1)
    def add(s, a, b): return D(max(s._l(a), s._l(b) if not isinstance(b, float) else 0))
    sub = add
    def inplace_add(s, a, b): a.d = max(a.d, s._l(b)); return a
    def scalar_sum(s, ts, cs): return D(max(s._l(t) for t in ts) + 1)


j = json.load(open(sys.argv[1]))
tot = [0, 0]
for name, c in j["softgelu"].items():
    cc, eps = [float(v) for v in c["cheb_coeffs"]], c.get("cheb_eps")
    o = Cnt(); d0 = poly.eval_chebyshev_bsgs(o, D(0), cc, -1.0, 1.0, eps).d; p0 = o.prod
    tr = poly._cheb_trim(cc, eps)
    ev = [tr[2 * k] for k in range(len(tr) // 2 + 1) if 2 * k < len(tr)]
    o = Cnt(); u = o.square(D(0))                   # u = 2 y^2 - 1: one square
    d1 = poly.eval_chebyshev_bsgs(o, u, ev, -1.0, 1.0, eps).d; p1 = o.prod
    tot[0] += p0; tot[1] += p1
    print(f"{name:26s} deg {len(cc)-1:3d}: y-series {p0:2d} products depth {d0} | u-series {p1:2d} products depth {d1}")
print(f"total products/token: y-series {tot[0]}, u-series {tot[1]}")

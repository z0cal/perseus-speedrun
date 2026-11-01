"""ct-ct products and depth of the GELU series: eval_chebyshev_bsgs (dead_ops) on y vs. the even part as a series in
u = T_2(y) = 2y^2 - 1 (T_2k(y) = T_k(u)) plus 0.5 y. usage (PYTHONPATH=python): python gelu_count.py CONFIG_JSON"""
import json, sys
import numpy as np
from perseus.impl import poly

class Cnt:
    cheb_bsgs = True; dead_ops = True; realize_shared = False
    def __init__(s): s.prod = 0
    def _l(s, a): return a if isinstance(a, int) else 0
    def square(s, a): s.prod += 1; return s._l(a) + 1
    def mult(s, a, b):
        if isinstance(b, float): return s._l(a) + 1
        s.prod += 1; return max(s._l(a), s._l(b)) + 1
    def add(s, a, b): return max(s._l(a), s._l(b) if not isinstance(b, float) else 0)
    sub = add
    def inplace_add(s, a, b): return a
    def scalar_sum(s, ts, cs): return max(ts) + 1

j = json.load(open(sys.argv[1]))
for name, c in j["softgelu"].items():
    cc = [float(v) for v in c["cheb_coeffs"]]
    o = Cnt(); d0 = poly.eval_chebyshev_bsgs(o, 0, cc, -1.0, 1.0); p0 = o.prod
    ev = [cc[2 * k] for k in range(len(cc) // 2 + 1) if 2 * k < len(cc)]
    o = Cnt(); u = o.square(0)                      # u = 2 y^2 - 1: one square
    d1 = poly.eval_chebyshev_bsgs(o, u, ev, -1.0, 1.0); p1 = o.prod
    print(f"{name:26s} deg {len(cc)-1:3d}: y-series {p0:2d} products depth {d0} | u-series {p1:2d} products depth {d1}")

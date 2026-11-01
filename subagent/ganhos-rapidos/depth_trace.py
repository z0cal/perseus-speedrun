"""Depth (levels) of Perseus's eval_chebyshev_bsgs on the shipped configs, with a tracer ops:
ct x ct product / square: max(depth) + 1; scalar product: depth + 1; adds: max; scalar_sum: max + 1."""
import json, math, os, sys
sys.path.insert(0, os.path.expanduser("~/pesquisa/perseus-speedrun/work/perseus-rec/python"))
from perseus.impl import poly

class D:
    def __init__(self, d): self.d = d
class Ops:
    cheb_bsgs = True; dead_ops = True; realize_shared = False; trace = False
    def __init__(self): self.nmul = 0; self.nsc = 0
    def _d(self, a): return a.d if isinstance(a, D) else -1
    def square(self, a): self.nmul += 1; return D(a.d + 1)
    def mult(self, a, b):
        if isinstance(b, D): self.nmul += 1; return D(max(a.d, b.d) + 1)
        self.nsc += 1; return D(a.d + 1)
    def add(self, a, b): return D(max(a.d, self._d(b)))
    def sub(self, a, b): return D(max(a.d, self._d(b)))
    def inplace_add(self, a, b): a.d = max(a.d, b.d)
    def realize(self, a): return a
    def scalar_sum(self, ts, cs): self.nsc += len(ts); return D(max(t.d for t in ts) + 1)
poly._LEAF_NATIVE = False

def depth(coeffs, a=-1.0, b=1.0, eps=None):
    o = Ops(); r = poly.eval_chebyshev_bsgs(o, D(0), list(coeffs), a, b, eps)
    return r.d, o.nmul, o.nsc

if __name__ == "__main__":
    W = os.path.expanduser("~/pesquisa/perseus-speedrun/work/perseus-rec/python/perseus/configs/model/approximation")
    c = json.load(open(f"{W}/gpt2_base_n32_kc16_cm/configs.json"))
    for k, g in c["softgelu"].items():
        cf = g["cheb_coeffs"]; tr = poly._cheb_trim(cf, g.get("cheb_eps"))
        print("GELU", k.split(".")[2], "deg", len(cf) - 1, "trimmed", len(tr) - 1, "depth,ctmul,scmul", depth(cf, -1, 1, g.get("cheb_eps")),
              "odd nz", sum(1 for i, x in enumerate(tr) if i % 2 and abs(x) > 0), "even nz", sum(1 for i, x in enumerate(tr) if i % 2 == 0 and abs(x) > 0))
    for k, n in sorted(c["norm"].items()):
        cf = n["cheb_coeffs"]
        print("LN", k, "deg", len(cf) - 1, "depth", depth(cf, n["cheb_lo"], n["cheb_hi"]))
    s = c["softmax"]["transformer.h.0.attn"]
    print("SM exp deg", len(s["cheb_coeffs"]) - 1, "depth", depth(s["cheb_coeffs"], -1, 1))
    for d in (3, 4, 5, 7, 8, 9, 13, 15, 16, 17, 27, 31, 32, 33, 59, 63, 64, 95, 119, 127, 128, 247, 255, 256):
        print("deg", d, "depth (dense coeffs)", depth([1.0] * (d + 1)))

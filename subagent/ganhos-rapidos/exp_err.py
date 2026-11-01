import json, math, sys, os
import numpy as np
from numpy.polynomial import chebyshev as C
sys.path.insert(0, "python")
from perseus.impl import poly
p = "python/perseus/configs/model/approximation/cand_%s/configs.json"
for name in sys.argv[1:]:
    c = json.load(open(p % name))
    for l in range(12):
        s = c["softmax"][f"transformer.h.{l}.attn"]
        a = s["cheb_b"]; cf = np.array(s["cheb_coeffs"])
        tr = poly._cheb_trim(list(cf))
        t = np.linspace(-1, 1, 20001)
        f = np.exp(a * t); g = C.chebval(t, tr)
        print(name, l, "l1", s["n_squarings"], "l2", s["refinement_iters"], "a %.3f" % a, "deg", len(cf) - 1, "trim", len(tr) - 1,
              "abs %.2e" % np.abs(g - f).max(), "rel2max %.2e" % (np.abs(g - f).max() / f.max()),
              "relmax %.2e" % np.abs((g - f) / f).max(), "squeeze", s["squeeze_bound"], "clip", s["clip_lo"], s["clip_hi"])

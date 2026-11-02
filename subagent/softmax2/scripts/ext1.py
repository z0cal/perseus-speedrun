"""Seeds with a low-end guard (fit_lp_ext): accuracy vs robustness below clip_lo."""
import sys
import numpy as np
import sm2, oor, pacote

CASES = {L: [(14, 1)] for L in (0, 1, 2, 9, 10)}

for L in [int(x) for x in sys.argv[1:]] or CASES:
    j, cfg = sm2.layer(L)
    W = j["clip_hi"] - j["clip_lo"]
    print(f"== camada {L} (W {W:.1f}, lo {j['clip_lo']:.1f}); recorde abaixo de lo: {oor.below(j, cfg, xs=(2, 5, 10, 20))}")
    for d, k1 in CASES[L]:
        for f in (0.0, 0.25, 0.5, 1.0):
            co, eps = sm2.fit_lp_ext(j, cfg, d, k1, f=f)
            if co is None:
                print(f"   ({d},{k1}) f={f}: inviavel"); continue
            c2 = sm2.with_seed(j, cfg, co, k1)
            m = sm2.evaluate(j, c2, N=2500, seed=11)["ALL"]
            c = dict(j); c.update(cheb_coeffs=list(co))
            print(f"   ({d},{k1}) guarda {f*W/2:5.1f} abaixo de lo: shape {m['shape']:.1e} maxdp {m['maxdp']:.1e} dout {m['dout']:.1e}"
                  f" sum|c| {np.abs(co).sum():.2f} | abaixo de lo: {oor.below(c, c2, xs=(2, 5, 10, 20))}", flush=True)

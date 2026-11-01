"""Sanity spot check of cand_rc3's round-0 bands on held-out windows, with the RUNTIME's divisor: inside
ref.softmax_ref (the encrypted path's float twin, SM_CHEB_GS=1) on the APPROXIMATE model's scores (kl_eval.forward_approx:
approx LN / GELU / softmax upstream), d = sum_k cheb_exp(s_k - mid)^(2^l1) / kc  (ref.py:85-99), vs band_check's exact
GPT-2 divisor (owt_windows.divisors) on the same windows. usage (work/perseus-rec, PYTHONPATH=python): band_spot.py N"""
import math, os, sys
import numpy as np
D = os.path.expanduser("~/pesquisa/perseus-speedrun")
sys.path.insert(0, f"{D}/scripts")
import kl_eval, owt_windows as ow  # noqa: E401,E402
from perseus.examples.gpt2 import ref  # noqa: E402
from perseus.impl import config as pconfig, smcheb  # noqa: E402
from perseus.impl.poly import eval_chebyshev  # noqa: E402
os.environ["SM_CHEB_GS"] = "1"
N = int(sys.argv[1])
CF = pconfig.load_configs(f"{D}/configs/cand_rc3/configs.json")
SM = [CF.softmax[f"transformer.h.{l}.attn"] for l in range(12)]
band = [smcheb.band_from_seed(s.init_alpha, s.init_beta) for s in SM]
docs, rnd = ow.offsets("holdout", N)
offs = docs[:N] + rnd[:N] + list(ow.STRESS)
mn_rt, mx_rt = [math.inf] * 12, [0.0] * 12
mn_ex, mx_ex = [math.inf] * 12, [0.0] * 12
bad_rt, bad_ex = set(), set()
orig = ref.softmax_ref
cur = {}
def tap(S, cfg, kc):
    l = SM.index(cfg)
    s = np.asarray(S, dtype=np.float64)
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    z = eval_chebyshev(ref._ops, s - mean, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = z * z
    d = (z / kc).sum(-1)
    mn_rt[l], mx_rt[l] = min(mn_rt[l], float(d.min())), max(mx_rt[l], float(d.max()))
    if (d < band[l][0]).any() or (d > band[l][1]).any():
        bad_rt.add(cur["o"])
    return orig(S, cfg, kc)
ref.softmax_ref = tap
for i, o in enumerate(offs):
    w = ow.windows([o])
    cur["o"] = o
    with np.errstate(all="ignore"):
        kl_eval.forward_approx(w[0], CF)
    Ss = ow.scores(w)
    for l in range(12):
        d = ow.divisors(Ss[l], (SM[l].clip_hi + SM[l].clip_lo) / 2, SM[l].log2delta2)
        mn_ex[l], mx_ex[l] = min(mn_ex[l], float(d.min())), max(mx_ex[l], float(d.max()))
        if (d < band[l][0]).any() or (d > band[l][1]).any():
            bad_ex.add(o)
print(f"windows {len(offs)} ({min(N, len(docs))} held-out doc starts + {N} held-out random + {len(ow.STRESS)} stress)")
for l in range(12):
    lo, hi = band[l]
    print(f"L{l:2d} l2={SM[l].log2delta2} band [{lo:.4g},{hi:.4g}] runtime-twin d [{mn_rt[l]:.4g},{mx_rt[l]:.4g}] "
          f"headroom {mn_rt[l] / lo:.2f}/{hi / mx_rt[l]:.2f} | exact d [{mn_ex[l]:.4g},{mx_ex[l]:.4g}] headroom "
          f"{mn_ex[l] / lo:.2f}/{hi / mx_ex[l]:.2f}")
print("outside (runtime twin):", sorted(bad_rt), " outside (exact):", sorted(bad_ex))
print(f"min headroom runtime twin: low {min(mn_rt[l] / band[l][0] for l in range(12)):.2f} high "
      f"{min(band[l][1] / mx_rt[l] for l in range(12)):.2f}; exact: low {min(mn_ex[l] / band[l][0] for l in range(12)):.2f} "
      f"high {min(band[l][1] / mx_ex[l] for l in range(12)):.2f}")

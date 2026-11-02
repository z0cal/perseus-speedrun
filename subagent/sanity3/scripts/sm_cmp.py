"""softmax_ref (upstream plaintext emulation of softmax_thor, with Goldschmidt / SM_CHEB_GS) vs the session's
sm_seed_sim.chain (exact division) vs exact softmax, on centred 16-key rows, record-4 config."""
import json, math, os, sys
os.environ.setdefault("SM_CHEB_GS", "1"); os.environ.setdefault("SM_GS_FIRST", "1")
import numpy as np
sys.path.insert(0, '/home/user/perseus-speedrun/scripts')
from sm_seed_sim import chain
from perseus.examples.gpt2.ref import softmax_ref, softmax_exact
from perseus.impl.config import parse_softmax, ref_key
CFG = sys.argv[1]
j = json.load(open(CFG))
rng = np.random.default_rng(1)
kc = 16
for L in [int(x) for x in sys.argv[2:]] or [0, 2, 5, 9]:
    s = j["softmax"][f"transformer.h.{L}.attn"]
    cfg = parse_softmax(s)
    mid, H, k1, d2 = s["mid_x"], s["squeeze_bound"], int(s["n_squarings"]), int(s["delta2"])
    lo_u, hi_u = s["clip_lo"], s["clip_hi"]
    N = 4000
    u = rng.uniform(lo_u, hi_u, (N, kc))
    u[: N // 2] = np.clip(rng.normal(rng.uniform(lo_u, hi_u, (N // 2, 1)), 2.0, (N // 2, kc)), lo_u, hi_u)
    head = 0
    r = ref_key(cfg.ref_keys[head], kc) if cfg.ref_keys else None
    if r is not None:
        u[:, r] = 0.0
    p_ex = softmax_exact(u)
    p_ref = np.real(softmax_ref(u, cfg, kc, head=head))
    t = (u - mid) / H
    p_sim = chain(lambda x: np.polynomial.chebyshev.chebval(x, np.array(s["cheb_coeffs"])), t, k1, d2)
    print(f"L{L} ref@{r} log2d1={cfg.log2delta1} log2d2(refinement_iters)={cfg.log2delta2} delta2={d2} sim_rounds={int(math.log2(d2))+1} "
          f"| max|p_ref-p_ex| {np.abs(p_ref - p_ex).max():.2e} | max|p_sim-p_ex| {np.abs(p_sim - p_ex).max():.2e} | max|p_ref-p_sim| {np.abs(p_ref - p_sim).max():.2e}"
          f" | sum p_ref range [{p_ref.sum(1).min():.6f},{p_ref.sum(1).max():.6f}]")
    pn = p_ref / p_ref.sum(1, keepdims=True)
    print(f"     renormalised p_ref: max|pn-p_ex| {np.abs(pn - p_ex).max():.2e}; max|pn-p_sim| {np.abs(pn - p_sim).max():.2e}; "
          f"row-scale err max {np.abs(p_ref.sum(1) - 1).max():.2e}; KL(p_ex||p_ref) mean {np.mean(np.sum(p_ex*np.log(p_ex/np.maximum(p_ref,1e-300)),1)):.2e}")

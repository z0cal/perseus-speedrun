"""Harness: record-4 softmax through the upstream float64 mirror (examples/gpt2/ref.py::softmax_ref, SM_CHEB_GS=1,
SM_GS_FIRST=1) on synthetic score rows, against the exact softmax. Seeds of the exp can be swapped in.
Run with PYTHONPATH=<clone>/python."""
import dataclasses, json, math, os
import numpy as np
from scipy.optimize import linprog
os.environ["SM_CHEB_GS"] = "1"; os.environ["SM_GS_FIRST"] = "1"
from perseus.examples.gpt2 import ref as R
from perseus.impl.config import parse_softmax, REF_SINK
from perseus.impl import smcheb

UP = os.environ.get("UP", os.path.expanduser("~/pesquisa/perseus-speedrun/work/perseus-r4"))   # a perseus clone (record 4)
CFG = json.load(open(f"{UP}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json"))


def layer(L):
    j = CFG["softmax"][f"transformer.h.{L}.attn"]
    return j, parse_softmax(j)


def head_sink(cfg):
    return next(h for h, r in enumerate(cfg.ref_keys) if r == REF_SINK) if cfg.ref_keys else None


def rows(j, kc, N, rng, kind, centred=True):
    lo, hi = j["clip_lo"], j["clip_hi"]
    if kind == "uniform":
        u = rng.uniform(lo, hi, (N, kc))
    elif kind == "cluster":
        u = rng.normal(rng.uniform(lo, hi, (N, 1)), 2.0, (N, kc))
    elif kind == "real":      # ref-dominant-ish rows: others N(mu, sigma), mu in [-8, 2]
        u = rng.normal(rng.uniform(-8, 2, (N, 1)), rng.uniform(0.5, 4, (N, 1)), (N, kc))
    elif kind == "flat":      # every key near the reference
        u = rng.normal(0.0, 0.3, (N, kc))
    elif kind == "top":       # keys near the top of the range
        u = rng.uniform(0.6 * hi, hi, (N, kc))
    elif kind == "bottom":    # keys at the bottom: head sum at its floor
        u = rng.uniform(lo, 0.9 * lo, (N, kc))
    else:
        raise ValueError(kind)
    u = np.clip(u, lo, hi)
    if centred:
        u[:, 0] = 0.0
    return u


def sm(cfg, u, kc):
    h = head_sink(cfg)
    return R.softmax_ref(u, cfg, kc, head=h) if h is not None else R.softmax_ref(u, cfg, kc)


def exact(u):
    e = np.exp(u - u.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def metrics(p, pe, V=None):
    dp = p - pe
    s = p.sum(-1)
    ps = p / s[:, None]
    m = {"maxdp": np.abs(dp).max(), "tv": np.abs(dp).sum(-1).max(), "tvmean": np.abs(dp).sum(-1).mean(),
         "fac": np.abs(s - 1).max(), "shape": np.abs(ps - pe).max(), "shapetv": np.abs(ps - pe).sum(-1).max()}
    if V is not None:
        o, oe = p @ V, pe @ V            # V [kc, 64]
        m["dout"] = (np.linalg.norm(o - oe, axis=-1) / np.sqrt(V.shape[1])).max()
        m["dout_mean"] = (np.linalg.norm(o - oe, axis=-1) / np.sqrt(V.shape[1])).mean()
    m["nan"] = int(np.sum(~np.isfinite(p)))
    return m


def with_seed(j, cfg, coeffs, k1):
    """cfg with the exp series `coeffs` (numpy Chebyshev on [-1, 1] of t, t = 2 ct / W) and k1 squarings,
    same clip range (same W, same delta2)."""
    W = j["clip_hi"] - j["clip_lo"]
    d2 = 2 ** cfg.log2delta2
    half = W / (2 * 2 ** k1 * d2)
    return dataclasses.replace(cfg, log2delta1=int(k1), cheb_coeffs=tuple(float(c) for c in coeffs),
                               cheb_a=-half, cheb_b=half)


def target_g(j, cfg, k1, u):
    """the exact g(u) = exp((u - U) / (delta1 delta2))."""
    d2 = 2 ** cfg.log2delta2
    return np.exp((u - j["clip_hi"]) / (2 ** k1 * d2))


def t_of(j, u):
    mean = 0.5 * (j["clip_hi"] + j["clip_lo"]); W = j["clip_hi"] - j["clip_lo"]
    return 2 * (u - mean) / W


def fit_interp(j, cfg, d, k1):
    """Chebyshev interpolant of g at d+1 Chebyshev points (near-minimax relative here: g = exp)."""
    u_of_t = lambda t: 0.5 * (j["clip_hi"] + j["clip_lo"]) + 0.5 * t * (j["clip_hi"] - j["clip_lo"])
    cheb = np.polynomial.chebyshev.Chebyshev.interpolate(lambda t: target_g(j, cfg, k1, u_of_t(t)), d)
    return cheb.coef


def band(j, cfg, k1, eps, u, nonneg=True, weight="mix"):
    n = 2 ** k1 * 2 ** cfg.log2delta2
    U = j["clip_hi"]
    e = np.exp(u)
    w = np.maximum(1.0, e) if weight == "mix" else e
    lo = np.maximum(e - eps * w, 0.0) ** (1.0 / n) * math.exp(-U / n)
    hi = (e + eps * w) ** (1.0 / n) * math.exp(-U / n)
    if not nonneg:
        lo = np.where(e - eps * w <= 0, -hi if k1 >= 1 else -hi, lo)
    return lo, hi


def lp_feasible(j, cfg, d, k1, eps, u, **kw):
    lo, hi = band(j, cfg, k1, eps, u, **kw)
    T = np.polynomial.chebyshev.chebvander(t_of(j, u), d)
    r = linprog(np.zeros(d + 1), A_ub=np.vstack([T, -T]), b_ub=np.r_[hi, -lo], bounds=[(None, None)] * (d + 1),
                method="highs")
    return (r.x if r.status == 0 else None)


def fit_lp(j, cfg, d, k1, eps=None, npts=1500, **kw):
    """weighted band LP; eps None: smallest feasible eps by bisection. returns (coeffs, eps)."""
    t = np.cos(np.pi * (np.arange(npts) + 0.5) / npts)
    u = 0.5 * (j["clip_hi"] + j["clip_lo"]) + 0.5 * t * (j["clip_hi"] - j["clip_lo"])
    u = np.r_[u, 0.0]
    if eps is not None:
        return lp_feasible(j, cfg, d, k1, eps, u, **kw), eps
    lo, hi = -12.0, 0.5
    best = lp_feasible(j, cfg, d, k1, 10 ** hi, u, **kw)
    if best is None:
        return None, float("inf")
    for _ in range(28):
        m = 0.5 * (lo + hi)
        x = lp_feasible(j, cfg, d, k1, 10 ** m, u, **kw)
        if x is not None:
            hi, best = m, x
        else:
            lo = m
    return best, 10 ** hi


def evaluate(j, cfg, N=4000, seed=0, kinds=("uniform", "cluster", "real", "flat", "top", "bottom"), kcs=(2, 4, 8, 16),
             nv=64, centred=None):
    centred = bool(cfg.ref_keys) if centred is None else centred
    rng = np.random.default_rng(seed)
    out = {}
    for kind in kinds:
        agg = {}
        for kc in kcs:
            u = rows(j, kc, N, rng, kind, centred)
            V = rng.normal(0, 1, (kc, nv))
            m = metrics(sm(cfg, u, kc), exact(u), V)
            for k, v in m.items():
                agg[k] = max(agg.get(k, 0), v)
        out[kind] = agg
    tot = {}
    for a in out.values():
        for k, v in a.items():
            tot[k] = max(tot.get(k, 0), v)
    out["ALL"] = tot
    return out


def fit_lp_ext(j, cfg, d, k1, f=0.5, npts=1500, nonneg=True):
    """fit_lp with the low end of the LP domain extended to t = -1 - f (same affine map, so the clip range and every
    other config key stay): below clip_lo the target is e^u (absolute criterion there), so the seed stays small and
    the squarings / rounds see ~0 instead of the Chebyshev tail."""
    t = np.cos(np.pi * (np.arange(npts) + 0.5) / npts)
    t = np.r_[t, -1.0 - f * (np.arange(1, 301) / 300.0)]
    u = 0.5 * (j["clip_hi"] + j["clip_lo"]) + 0.5 * t * (j["clip_hi"] - j["clip_lo"])
    u = np.r_[u, 0.0]
    lo, hi = -12.0, 0.5
    best = lp_feasible(j, cfg, d, k1, 10 ** hi, u, nonneg=nonneg)
    if best is None:
        return None, float("inf")
    for _ in range(28):
        m = 0.5 * (lo + hi)
        x = lp_feasible(j, cfg, d, k1, 10 ** m, u, nonneg=nonneg)
        if x is not None:
            hi, best = m, x
        else:
            lo = m
    return best, 10 ** hi

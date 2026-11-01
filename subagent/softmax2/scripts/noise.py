"""Replica of ref.softmax_ref (SM_CHEB_GS=1) with refresh noise: absolute Gaussian noise of std sigma on the exp's
output z (SB#1, before the active mask), and optionally on each round's output y (SB#2, SB#3...). Reports the
amplification rms|dp| / sigma and max|dp| / sigma, plus the denominators seen by each round against its band."""
import math
import numpy as np
import sm2
from perseus.examples.gpt2 import ref as R
from perseus.impl import smcheb
from perseus.impl.poly import goldschmidt_recip


def run(cfg, u, kc, sigma=0.0, where=("exp",), rng=None, track=None):
    h = sm2.head_sink(cfg)
    s = np.asarray(u, float)
    if cfg.ref_keys:
        s = s - s[..., R.ref_key(cfg.ref_keys[h], kc), None]
    mean = 0.5 * (cfg.clip_hi + cfg.clip_lo)
    ct = s - mean
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    z = R.eval_chebyshev(R._ops, ct, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = z * z
    if sigma and "exp" in where:
        z = z + sigma * rng.normal(size=z.shape)
    z = z / kc
    y = None
    rounds = smcheb.rounds(cfg, kc)
    for k, (alpha, beta, iters, scale, cheb) in enumerate(rounds):
        if k:
            z = (2.0 * y) ** 2
        d = z.sum(axis=-1, keepdims=True) * np.ones_like(z)
        if track is not None:
            track.setdefault(k, []).append((float((scale * d).min()), float((scale * d).max())))
        r = goldschmidt_recip(R._ops, d, alpha, beta, iters, scale, Dh=0.5 * beta * scale * scale * d, cheb=cheb)
        y = z * np.real(r)
        if sigma and ("rounds" in where or ("last" in where and k == len(rounds) - 1)):
            y = y + sigma * rng.normal(size=y.shape)
    return y


def bands(cfg, kc):
    out = []
    lo, hi = smcheb.band_from_seed(cfg.init_alpha, cfg.init_beta)
    out.append((lo, hi))
    for j in range(1, len(cfg.recip_iters_cheb)):
        out.append(smcheb.refine_band(kc, float(cfg.recip_targets[j - 1])))
    return out


def amp(j, cfg, kinds=("real", "uniform", "flat", "bottom"), kcs=(2, 4, 8, 16), N=2000, sigma=1e-7, where=("exp",),
        seed=3):
    rng = np.random.default_rng(seed)
    res = {}
    for kind in kinds:
        a_rms, a_max = 0.0, 0.0
        for kc in kcs:
            u = sm2.rows(j, kc, N, rng, kind, bool(cfg.ref_keys))
            p0 = run(cfg, u, kc)
            p1 = run(cfg, u, kc, sigma, where, rng)
            dp = p1 - p0
            a_rms = max(a_rms, float(np.sqrt((dp ** 2).mean())) / sigma)
            a_max = max(a_max, float(np.abs(dp).max()) / sigma)
        res[kind] = (a_rms, a_max)
    return res

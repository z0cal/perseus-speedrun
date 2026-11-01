"""H3: layer 4 (and a generic centred layer) rebuilt with fit_gpt2_decode16_ref.fit_centred's formulas for a chosen
delta2 / margin / delta1 / exp degree; depth (the script's depth()), band, accuracy (softmax_ref) and exp-refresh
noise amplification (noise.amp)."""
import dataclasses, json, math, sys
import numpy as np
import sm2, noise
from perseus.impl import smcheb
from perseus.impl.config import parse_softmax
KMAX = 16


def next_pow2(x):
    return 1 if x <= 1 else 1 << math.ceil(math.log2(x))


def cheb_exp_series(lo, hi, degree, min_nodes=200):
    n = max(8 * (degree + 1), min_nodes)
    k = np.arange(1, n + 1)
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos((2 * k - 1) * np.pi / (2 * n))
    series = np.polynomial.chebyshev.Chebyshev.fit(x, np.exp(x), degree, domain=[lo, hi])
    return (series.coef / float(series(0.0))).astype(np.float64)


def build(base, ustar, ulo, margin=1.25, delta2=None, delta1=None, degree=8, tau=4.6, d0=5.0, lo_margin=0.05,
          tgt=1e-3):
    U = margin * ustar
    lo = ulo - lo_margin * (U - ulo)
    W = U - lo
    delta2 = delta2 or next_pow2(ustar / tau)
    delta1 = delta1 or max(next_pow2(W / (d0 * delta2)), 1)
    delta0 = W / (delta1 * delta2)
    coeffs = cheb_exp_series(-0.5 * delta0, 0.5 * delta0, degree) * math.exp(-0.5 * delta0)
    band = (math.exp(-U / delta2) / KMAX, 1.05)
    ia, ib = smcheb.cheb_seed(*band)
    R = int(round(math.log2(delta2)))
    ks, tg = smcheb.design(ia, ib, R, tgt_last=tgt, kcs=tuple(range(1, KMAX + 1)))
    refine = smcheb.cheb_seed(*smcheb.refine_band(KMAX, 0.3))
    c = dict(base)
    c.update({"mid_x": 0.5 * (lo + U), "input_scale": float(delta1 * delta2), "squeeze_bound": 0.5 * W,
              "clip_lo": lo, "clip_hi": U, "n_squarings": int(round(math.log2(delta1))), "refinement_iters": R,
              "cheb_coeffs": coeffs.tolist(), "cheb_a": -0.5 * delta0, "cheb_b": 0.5 * delta0,
              "init_alpha": ia, "init_beta": ib, "refine_alpha": [refine[0]] * R, "refine_beta": [refine[1]] * R,
              "gs_iters_scaled": int(ks[0]), "gs_iters_first": int(ks[0]),
              "gs_iters_refine_scaled": int(max(ks[1:], default=0)), "per_step_refine_iters": [int(k) for k in ks[1:]],
              "sm_kc_r": [1.0] * (R * KMAX), "recip_iters_cheb": [int(k) for k in ks],
              "recip_targets": [float(t) for t in tg], "delta0": delta0, "delta1": int(delta1), "delta2": int(delta2)})
    return c


def depth(c, exp_levels=4):
    k = [int(x) for x in c["recip_iters_cheb"]]
    return exp_levels + int(c["n_squarings"]) + k[0] + 2 + sum(x + 3 for x in k[1:])


def report(name, c, exp_levels=4, N=2000):
    cfg = parse_softmax(c)
    m = sm2.evaluate(c, cfg, N=N, seed=5)["ALL"]
    a = noise.amp(c, cfg, N=800)
    amax = max(v[1] for v in a.values()); arms = max(v[0] for v in a.values())
    print(f"{name:34s} U {c['clip_hi']:.2f} d1 {c['delta1']} d2 {c['delta2']} d0 {c['delta0']:.2f} counts "
          f"{c['recip_iters_cheb']} alvos {[round(t, 3) for t in c['recip_targets']]} profundidade {depth(c, exp_levels)} | "
          f"banda0 piso {math.exp(-c['clip_hi'] / c['delta2']) / 16:.2e} | maxdp {m['maxdp']:.1e} shape {m['shape']:.1e} "
          f"dout {m['dout']:.1e} NaN {m['nan']} | ruido exp rms x{arms:.0f} max x{amax:.0f}", flush=True)
    return c


if __name__ == "__main__":
    j, _ = sm2.layer(4)
    us, ul = 18.57, -96.88
    rec = report("recorde (json)", dict(j))
    report("refeito tau 4.6 (confere)", build(j, us, ul))
    report("tau 5.0 (d2 4, d1 auto)", build(j, us, ul, tau=5.0))
    report("d2 4, d1 4, grau 8", build(j, us, ul, delta2=4, delta1=4))
    report("d2 4, d1 4, grau 12", build(j, us, ul, delta2=4, delta1=4, degree=12))
    for mg in (1.15, 1.10, 1.05, 1.0):
        report(f"d2 4 margem {mg} d1 auto", build(j, us, ul, margin=mg, delta2=4))
        report(f"d2 4 margem {mg} d1 4 grau 12", build(j, us, ul, margin=mg, delta2=4, delta1=4, degree=12))

"""Candidate 1 of the softmax review: layer 4 of record 4 with delta2 = 4 instead of 8 (one THOR round less).

1. Rebuilds layer 4 with fit_gpt2_decode16_ref.fit_centred's formulas from the record config's own statistics
   (U* = clip_hi / 1.25, u_lo from clip_lo) and checks it against the shipped layer (so the candidates below differ
   from the record only where they should).
2. Builds candidates with delta2 forced to 4 and margin U / U* in {1.25 (= the upstream's rejected tau 5.0), 1.1,
   1.05, 1.0}.
3. Measures, on synthetic 16-key rows through the upstream float64 mirror (examples/gpt2/ref.py::softmax_ref,
   SM_CHEB_GS=1, SM_GS_FIRST=1):
   - accuracy against the exact softmax (max|dp|, TV, attention-output error with a random V);
   - refresh-noise amplification: absolute Gaussian noise on the exp's output (the first SparseB refresh), rms|dp|/sigma,
     next to the same number for every record layer (the record's worst layer is the reference: the upstream rejected
     tau 5.0 because a refresh-noise model's KL doubled);
   - tolerance above U*: the largest score x U* a single key may reach before the Goldschmidt leaves its band and the
     output stops being a softmax (|p - p_exact| > 0.01 or NaN).
4. Writes full configs.json variants (record config, layer 4 replaced) to OUT_DIR/<name>/configs.json.

usage: PYTHONPATH=<perseus clone>/python python sm_l4_delta2.py [OUT_DIR]   (numpy + scipy; ~3 min)"""
import copy
import dataclasses
import json
import math
import os
import sys

import numpy as np

os.environ.setdefault("SM_CHEB_GS", "1")
os.environ.setdefault("SM_GS_FIRST", "1")
from perseus.examples.gpt2 import ref as R  # noqa: E402
from perseus.impl import smcheb  # noqa: E402
from perseus.impl.config import parse_softmax, REF_SINK  # noqa: E402
from perseus.impl.poly import goldschmidt_recip  # noqa: E402

import perseus  # noqa: E402
UP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(perseus.__file__))))
CFG_PATH = f"{UP}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json"
KMAX = 16
L4 = "transformer.h.4.attn"


# ── fit_gpt2_decode16_ref.fit_centred, from statistics instead of score rows ──
def next_pow2(x):
    return 1 if x <= 1 else 1 << math.ceil(math.log2(x))


def cheb_exp_series(lo, hi, degree, min_nodes=200):
    """fit_gpt2_decode16.cheb_exp_series."""
    n = max(8 * (degree + 1), min_nodes)
    k = np.arange(1, n + 1)
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos((2 * k - 1) * np.pi / (2 * n))
    series = np.polynomial.chebyshev.Chebyshev.fit(x, np.exp(x), degree, domain=[lo, hi])
    return (series.coef / float(series(0.0))).astype(np.float64)


def build(base, ustar, ulo, margin=1.25, delta2=None, tau=4.6, d0=5.0, lo_margin=0.05, tgt=1e-3):
    U = margin * ustar
    lo = ulo - lo_margin * (U - ulo)
    W = U - lo
    delta2 = delta2 or next_pow2(ustar / tau)
    delta1 = max(next_pow2(W / (d0 * delta2)), 1)
    delta0 = W / (delta1 * delta2)
    coeffs = cheb_exp_series(-0.5 * delta0, 0.5 * delta0, 8) * math.exp(-0.5 * delta0)
    band = (math.exp(-U / delta2) / KMAX, 1.05)
    ia, ib = smcheb.cheb_seed(*band)
    Rn = int(round(math.log2(delta2)))
    ks, tg = smcheb.design(ia, ib, Rn, tgt_last=tgt, kcs=tuple(range(1, KMAX + 1)))
    refine = smcheb.cheb_seed(*smcheb.refine_band(KMAX, 0.3))
    c = dict(base)
    c.update({"mid_x": 0.5 * (lo + U), "input_scale": float(delta1 * delta2), "squeeze_bound": 0.5 * W,
              "clip_lo": lo, "clip_hi": U, "n_squarings": int(round(math.log2(delta1))), "refinement_iters": Rn,
              "cheb_coeffs": coeffs.tolist(), "cheb_a": -0.5 * delta0, "cheb_b": 0.5 * delta0,
              "init_alpha": ia, "init_beta": ib, "refine_alpha": [refine[0]] * Rn, "refine_beta": [refine[1]] * Rn,
              "gs_iters_scaled": int(ks[0]), "gs_iters_first": int(ks[0]),
              "gs_iters_refine_scaled": int(max(ks[1:], default=0)), "per_step_refine_iters": [int(k) for k in ks[1:]],
              "sm_kc_r": [1.0] * (Rn * KMAX), "recip_iters_cheb": [int(k) for k in ks],
              "recip_targets": [float(t) for t in tg], "delta0": delta0, "delta1": int(delta1), "delta2": int(delta2)})
    c["poly_coeffs"] = np.polynomial.chebyshev.cheb2poly(coeffs).tolist()
    c["note"] = (base.get("note", "") + f" | sm_l4_delta2.py: delta2 {delta2}, margin {margin:g} (U {U:.3f}, "
                 f"U* {ustar:.3f})")
    return c


def depth(c):
    """fit_gpt2_decode16_ref.depth: exp (4) + squarings + per round its count, the head-sum mask and the product."""
    k = c["recip_iters_cheb"]
    return 4 + int(c["n_squarings"]) + k[0] + 2 + sum(int(x) + 3 for x in k[1:])


# ── the float64 mirror on synthetic rows ──
def head_sink(cfg):
    return next(h for h, r in enumerate(cfg.ref_keys) if r == REF_SINK) if cfg.ref_keys else None


def rows(j, kc, N, rng, kind):
    lo, hi = j["clip_lo"], j["clip_hi"]
    if kind == "uniform":
        u = rng.uniform(lo, hi, (N, kc))
    elif kind == "cluster":
        u = rng.normal(rng.uniform(lo, hi, (N, 1)), 2.0, (N, kc))
    elif kind == "real":
        u = rng.normal(rng.uniform(-8, 2, (N, 1)), rng.uniform(0.5, 4, (N, 1)), (N, kc))
    elif kind == "flat":
        u = rng.normal(0.0, 0.3, (N, kc))
    elif kind == "top":
        u = rng.uniform(0.6 * hi, hi, (N, kc))
    elif kind == "bottom":
        u = rng.uniform(lo, 0.9 * lo, (N, kc))
    u = np.clip(u, lo, hi)
    u[:, 0] = 0.0                       # the reference key (centred layer)
    return u


def exact(u):
    e = np.exp(u - u.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def accuracy(j, N=3000, seed=5, kinds=("uniform", "cluster", "real", "flat", "top", "bottom"), kcs=(2, 4, 8, 16)):
    cfg = parse_softmax(j)
    rng = np.random.default_rng(seed)
    m = dict(maxdp=0.0, tv=0.0, dout=0.0, nan=0)
    for kind in kinds:
        for kc in kcs:
            u = rows(j, kc, N, rng, kind)
            p = R.softmax_ref(u, cfg, kc, head=head_sink(cfg))
            pe = exact(u)
            V = rng.normal(0, 1, (kc, 64))
            dp = p - pe
            m["maxdp"] = max(m["maxdp"], float(np.nanmax(np.abs(dp))))
            m["tv"] = max(m["tv"], float(np.nanmax(np.abs(dp).sum(-1))))
            m["dout"] = max(m["dout"], float(np.nanmax(np.linalg.norm(dp @ V, axis=-1) / 8.0)))
            m["nan"] += int(np.sum(~np.isfinite(p)))
    return m


def mirror_noisy(cfg, u, kc, sigma, rng):
    """softmax_ref (SM_CHEB_GS=1) with absolute Gaussian noise of std sigma on the exp's output (first refresh)."""
    s = np.asarray(u, float)
    h = head_sink(cfg)
    if cfg.ref_keys:
        s = s - s[..., R.ref_key(cfg.ref_keys[h], kc), None]
    ct = s - 0.5 * (cfg.clip_hi + cfg.clip_lo)
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    z = R.eval_chebyshev(R._ops, ct, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = z * z
    if sigma:
        z = z + sigma * rng.normal(size=z.shape)
    z = z / kc
    y = None
    for k, (alpha, beta, iters, scale, cheb) in enumerate(smcheb.rounds(cfg, kc)):
        if k:
            z = (2.0 * y) ** 2
        d = z.sum(axis=-1, keepdims=True) * np.ones_like(z)
        r = goldschmidt_recip(R._ops, d, alpha, beta, iters, scale, Dh=0.5 * beta * scale * scale * d, cheb=cheb)
        y = z * np.real(r)
    return y


def noise_amp(j, N=1500, sigma=1e-7, seed=3, kinds=("real", "uniform", "flat", "bottom"), kcs=(2, 4, 8, 16)):
    """max over row kinds and kc of rms|dp| / sigma (the noise model's amplification of the first refresh)."""
    cfg = parse_softmax(j)
    rng = np.random.default_rng(seed)
    a = 0.0
    for kind in kinds:
        for kc in kcs:
            u = rows(j, kc, N, rng, kind) if cfg.ref_keys else rows_uncentred(j, kc, N, rng, kind)
            p0 = mirror_noisy(cfg, u, kc, 0.0, rng)
            p1 = mirror_noisy(cfg, u, kc, sigma, rng)
            a = max(a, float(np.sqrt(((p1 - p0) ** 2).mean())) / sigma)
    return a


def rows_uncentred(j, kc, N, rng, kind):
    u = rows(j, kc, N, rng, kind)
    u[:, 0] = rng.uniform(j["clip_lo"], j["clip_hi"], N)
    return u


def tolerance(j, ustar, kcs=(1, 2, 4, 16)):
    """Largest x (in units of U*) such that one key at x U* (the others at the reference, 0) still gives
    |p - p_exact| <= 0.01 and finite output, for every kc."""
    cfg = parse_softmax(j)
    worst = float("inf")
    for kc in kcs:
        last_ok = 0.0
        for x in np.arange(1.0, 2.5, 0.01):
            u = np.zeros((1, kc))
            u[0, -1] = x * ustar
            u[0, 0] = 0.0
            p = R.softmax_ref(u, cfg, kc, head=head_sink(cfg))
            if not np.all(np.isfinite(p)) or np.abs(p - exact(u)).max() > 0.01:
                break
            last_ok = x
        worst = min(worst, last_ok)
    return worst


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None
    cfg_all = json.load(open(CFG_PATH))
    j4 = cfg_all["softmax"][L4]
    ustar = j4["clip_hi"] / 1.25
    ulo = (j4["clip_lo"] + 0.05 * j4["clip_hi"]) / 1.05
    print(f"camada 4 do recorde: U* = {ustar:.4f}, u_lo = {ulo:.4f}, delta2 = {j4['delta2']} "
          f"(U*/4.6 = {ustar / 4.6:.3f}; delta2 = 4 pede U* <= {4 * 4.6:.2f})")

    rebuilt = build(j4, ustar, ulo)
    keys = [k for k in j4 if k not in ("note",)]
    diff = {k: (j4[k], rebuilt[k]) for k in keys if k in rebuilt and json.dumps(j4[k]) != json.dumps(rebuilt[k])}
    num = {k: v for k, v in diff.items()
           if isinstance(v[0], (int, float, list)) and np.allclose(np.asarray(v[0], float), np.asarray(v[1], float),
                                                                   rtol=1e-12, atol=1e-14)}
    print(f"reconstrução: {len(keys)} chaves, {len(diff) - len(num)} diferentes "
          f"({len(num)} só em ~1e-15)" + ("" if len(diff) == len(num) else f": {sorted(set(diff) - set(num))}"))

    print("\nruído do 1º refresh (rms|dp|/sigma), camadas do recorde:")
    rec_amp = {}
    for L in range(12):
        jl = cfg_all["softmax"][f"transformer.h.{L}.attn"]
        rec_amp[L] = noise_amp(jl)
    print("   " + "  ".join(f"L{L} {a:.0f}" for L, a in rec_amp.items()))
    worst_L = max((L for L in rec_amp if cfg_all["softmax"][f"transformer.h.{L}.attn"].get("ref_keys")),
                  key=lambda L: rec_amp[L])
    print(f"   pior camada centrada: L{worst_L} ({rec_amp[worst_L]:.0f})")

    cands = [("recorde (delta2 8, margem 1,25)", j4)]
    for mg in (1.25, 1.10, 1.05, 1.00):
        cands.append((f"delta2 4, margem {mg:.2f}", build(j4, ustar, ulo, margin=mg, delta2=4)))
    print(f"\n{'candidata':34s} {'d1':>3s} {'d2':>3s} {'contagens GS':18s} {'prof.':>5s} {'maxdp':>8s} {'TV':>8s} "
          f"{'dout':>8s} {'NaN':>4s} {'ruído':>6s} {'xL' + str(worst_L):>6s} {'tol./U*':>7s}")
    for name, c in cands:
        acc = accuracy(c)
        a = noise_amp(c)
        tol = tolerance(c, ustar)
        print(f"{name:34s} {c['delta1']:3d} {c['delta2']:3d} {str(c['recip_iters_cheb']):18s} {depth(c):5d} "
              f"{acc['maxdp']:8.1e} {acc['tv']:8.1e} {acc['dout']:8.1e} {acc['nan']:4d} {a:6.0f} "
              f"{a / rec_amp[worst_L]:6.2f} {tol:7.2f}")
        if out_dir and c is not j4:
            tag = name.split("margem ")[1].replace(",", "").replace(".", "")
            d = os.path.join(out_dir, f"r4_l4d4_m{tag}")
            os.makedirs(d, exist_ok=True)
            full = copy.deepcopy(cfg_all)
            full["softmax"][L4] = c
            json.dump(full, open(os.path.join(d, "configs.json"), "w"), indent=2)
    if out_dir:
        print(f"\nconfigs gravadas em {out_dir}/r4_l4d4_m*/configs.json (recorde 4 com a camada 4 trocada)")


if __name__ == "__main__":
    main()

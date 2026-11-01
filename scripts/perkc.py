"""Per-kc round-0 band for the Decode-8 record's softmax division (SM_CHEB_GS), simulated in float64 with Perseus's own
reference primitives (examples/gpt2/ref.py: layer_norm_ref, softmax_ref; the Chebyshev GELU as GELU_FOLD runs it).

Round 0 divides by d = sum_j z_j, z = exp((x - mid) / delta2) / kc. The record seeds it on one band for every kc
(band_from_seed(init_alpha, init_beta)); here each kc gets its own band, the data's [min, max] of d at that kc over
the calibration prompts, widened by MARGIN on both ends and kept inside the record's band, with its own minimax seed.
Round 0's count becomes the max over kc of the count that band needs for the round's target; the refine rounds are
the record's. KL(exact float64 GPT-2 || approximate model) of the logits at positions 8..15 (CutMax left out).

usage: PYTHONPATH=<perseus>/python python -I perkc.py PROMPTS_JSON CONFIG_JSON [MARGIN] [N_CAL]"""
import json
import math
import os
import sys

os.environ["SM_CHEB_GS"] = "1"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import exp  # noqa: E402
from perseus.examples.gpt2 import ref  # noqa: E402
from perseus.impl import config as pconfig  # noqa: E402
from perseus.impl import smcheb  # noqa: E402
from perseus.impl.poly import eval_chebyshev  # noqa: E402

W, L, H, D, dh = exp.W, exp.L, exp.H, exp.D, exp.dh
T = 16
prompts = json.load(open(sys.argv[1]))["prompts"]
CF = pconfig.load_configs(sys.argv[2])
margin = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
n_cal = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SM = [CF.softmax[f"transformer.h.{l}.attn"] for l in range(L)]
_ops = ref._ops


def gelu_cheb(x, cfg):
    return cfg.xmax * eval_chebyshev(_ops, x / cfg.xmax, cfg.cheb_coeffs, cfg.cheb_a, cfg.cheb_b)


def forward_approx(toks, tap=None):
    """Teacher-forced decode of positions 0..T-1 on the approximations; `tap(l, kc, scores)` sees each softmax input."""
    x = W["wte.weight"][toks] + W["wpe.weight"][: len(toks)]
    Ks = [[] for _ in range(L)]
    Vs = [[] for _ in range(L)]
    out = []
    for t in range(len(toks)):
        h = x[t].copy()
        for l in range(L):
            p = f"h.{l}."
            a = ref.layer_norm_ref(h, CF.norm[f"transformer.h.{l}.ln_1"], t, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
            q, k, v = np.split(a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"], 3)
            Ks[l].append(k)
            Vs[l].append(v)
            K, V = np.array(Ks[l]).reshape(t + 1, H, dh), np.array(Vs[l]).reshape(t + 1, H, dh)
            S = np.einsum("hd,khd->hk", q.reshape(H, dh), K) / math.sqrt(dh)
            if tap:
                tap(l, t + 1, S)
            P = ref.softmax_ref(S, SM[l], t + 1)
            o = np.einsum("hk,khd->hd", P, V).reshape(D)
            h = h + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
            a = ref.layer_norm_ref(h, CF.norm[f"transformer.h.{l}.ln_2"], t, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
            g = gelu_cheb(a @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"], CF.gelu[f"transformer.h.{l}.mlp.act"])
            h = h + g @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        f = CF.norm["transformer.ln_f"]
        out.append(ref.layer_norm_ref(h, f, t, W["ln_f.weight"], W["ln_f.bias"]) @ W["wte.weight"].T)
    return np.array(out)


def round0_d(cfg, kc, S):
    """The round-0 divisor softmax_ref computes (exact exp in place of its Chebyshev + squarings)."""
    mid = (cfg.clip_hi + cfg.clip_lo) / 2.0
    return np.exp((S - mid) / 2 ** cfg.log2delta2).sum(-1) / kc


def run(test, label):
    kls = []
    for toks in test:
        r = exp.forward(np.array(toks[:T]), exp.sm_exact)
        a = forward_approx(np.array(toks[:T]))
        kls.append(exp.kl(r, a)[8:T])
    k = np.concatenate(kls)
    bad = ~np.isfinite(k)
    print(f"{label:34s} KL pos 8-15: mean {np.nanmean(k):.3e}  median {np.nanmedian(k):.3e}  max {np.nanmax(k):.3e}"
          f"  non-finite {bad.sum()}/{k.size} (prompts {sorted(set(np.nonzero(bad)[0] // 8))})")
    return k


def main():
    global rng, band, newk, viol, cal, test
    # ---- calibration: per-(layer, kc) range of the round-0 divisor on the calibration prompts
    cal, test = prompts[len(prompts) - n_cal:], prompts[: len(prompts) - n_cal]
    rng = {}


    def tap(l, kc, S):
        d = round0_d(SM[l], kc, S)
        lo, hi = rng.get((l, kc), (np.inf, 0.0))
        rng[(l, kc)] = (min(lo, d.min()), max(hi, d.max()))


    for toks in cal:
        forward_approx(np.array(toks[:T]), tap)

    # test rows' divisors against both bands (outside a band = the seed's guarantee is gone)
    viol = {"record": 0, "per-kc": 0, "rows": 0}
    band = {}
    print(f"calibration on {len(cal)} prompts, test on {len(test)}, margin x{margin:g}\n")
    print(f"{'L':>2} {'record band':>22} {'k0 rec':>6} {'k0 per-kc':>9}   worst kc band")
    newk = []
    for l in range(L):
        cfg = SM[l]
        glo, ghi = smcheb.band_from_seed(cfg.init_alpha, cfg.init_beta)
        tg = float(cfg.recip_targets[0])
        ks = {}
        for kc in range(1, T + 1):
            lo, hi = rng[(l, kc)]
            lo, hi = max(lo / margin, glo), min(hi * margin, ghi)
            band[(l, kc)] = (lo, hi)
            ks[kc] = smcheb.k_for(lo, hi, tg)
        k0 = max(ks.values())
        newk.append(k0)
        w = max(ks, key=ks.get)
        print(f"{l:>2} [{glo:9.3g}, {ghi:9.3g}] {cfg.recip_iters_cheb[0]:>6} {k0:>9}   kc={w} "
              f"[{band[(l, w)][0]:.3g}, {band[(l, w)][1]:.3g}]")
    print(f"round-0 levels per token: record {sum(c.recip_iters_cheb[0] for c in SM)}, per-kc {sum(newk)}\n")


    def tap_test(l, kc, S):
        d = round0_d(SM[l], kc, S)
        glo, ghi = smcheb.band_from_seed(SM[l].init_alpha, SM[l].init_beta)
        lo, hi = band[(l, kc)]
        viol["rows"] += d.size
        viol["record"] += int(((d < glo) | (d > ghi)).sum())
        bad = (d < lo) | (d > hi)
        viol["per-kc"] += int(bad.sum())
        for v in d[bad]:
            print(f"  outside: layer {l} kc {kc} d={v:.4g} band [{lo:.4g}, {hi:.4g}] cal range {rng[(l, kc)]}")


    for toks in test:
        forward_approx(np.array(toks[:T]), tap_test)
    print(f"test rows outside the band: record {viol['record']}/{viol['rows']}, per-kc {viol['per-kc']}/{viol['rows']}\n")

    base = run(test, "record (baseline)")

    orig_rounds = smcheb.rounds


    def rounds_perkc(cfg, kc):
        out = orig_rounds(cfg, kc)
        l = SM.index(cfg)
        lo, hi = band[(l, kc)]
        k = newk[l]
        a, b = smcheb.cheb_seed(lo, hi)
        kap, k0, mu = smcheb.step_consts(smcheb.z0_of(lo, hi), k)
        out[0] = (a * k0, b * k0, k, 1.0, (kap, mu))
        return out


    smcheb.rounds = rounds_perkc
    new = run(test, "per-kc round 0")
    ratio = (new / base)[np.isfinite(new)]
    print(f"\nper-kc / record KL per position (finite): geo-mean {np.exp(np.log(ratio).mean()):.3f}, "
          f"max {ratio.max():.3f}; bar 1e-3")


if __name__ == "__main__":
    main()

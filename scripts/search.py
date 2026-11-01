"""Training-free search over the Decode-8 approximations (ATLAS-style, stage 1: uniform knobs; stage 2: per layer).

A candidate is the record config (gpt2_base_n32_kc16) with some approximations refit at a lower degree:
  gelu=D        every GELU Chebyshev series of degree > D refit at degree D (on the same xmax)
  gelu.L=D      one layer's
  ln=D / ln.KEY=D   LayerNorm inverse square roots (inv_out_scale / sqrt on [cheb_lo, cheb_hi])
  smtgt=E       softmax: the last reciprocal round's target E (recip_iters_cheb redesigned by impl.smcheb.design)
  smtop=M       softmax: round 0's band top at M x the calibration max of its divisor (round0_ranges)
  from=NAME     start from cand_NAME's config instead of the record's
  defer=L,L     softmax_thor_deferred (defer_norm) in these layers (needs the defer-norm branch)
Each candidate: KL on owt windows (kl_eval), capture under the record runtime (capture_rec.sh, record worktree), and
blocks 0-12 bootstraps under the em9 recipe (plan_count.sh; min-cut proxy, or ILP with --ilp).

    python -I search.py NAME KNOB [KNOB ...] [--ilp] [--no-plan]
Results are appended to ~/pesquisa/perseus-speedrun/work/search.tsv.
"""
import copy
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev as C

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.expanduser("~/pesquisa/perseus-speedrun/work")
REC = f"{W}/perseus-rec"
BASE = f"{REC}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16/configs.json"
sys.path.insert(0, HERE)
sys.path.insert(0, f"{REC}/python")


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def cheb_fit(f, deg, n=None):
    """Near-minimax: least squares on many Chebyshev nodes of [-1, 1] (numpy convention, c0 not halved)."""
    t = np.cos(np.pi * (np.arange(n or 8 * (deg + 1)) + 0.5) / (n or 8 * (deg + 1)))
    return C.chebfit(t, f(t), deg)


def refit_gelu(g, deg):
    X = g["xmax"]
    g["cheb_coeffs"] = list(cheb_fit(lambda t: gelu(X * t) / X, deg))
    g["note"] = f"refit degree {deg} (search.py)"


def refit_ln(n, deg):
    lo, hi, s = n["cheb_lo"], n["cheb_hi"], n["inv_out_scale"]
    n["cheb_coeffs"] = list(cheb_fit(lambda t: s / np.sqrt(lo + (hi - lo) * (t + 1) / 2), deg))
    n["note"] = f"refit degree {deg} (search.py)"


def retarget_softmax(s, tgt_last):
    from perseus.impl import smcheb
    from perseus.impl.config import parse_softmax
    cfg = parse_softmax(s)
    ks, tg = smcheb.design(cfg.init_alpha, cfg.init_beta, cfg.log2delta2, tgt_last,
                           tuple(k for k in smcheb.KC_DESIGN if k <= 16) + (16,))
    s["recip_iters_cheb"], s["recip_targets"] = ks, tg


def round0_ranges():
    """Per layer, the [min, max] of softmax round 0's divisor d = sum_j exp((x_j - mid) / delta2) / kc over 256
    calibration windows of the pool (offsets 1000000 + 3889 i, i < 256: disjoint from kl_eval's and the record's
    rows); cached."""
    path = f"{W}/round0_ranges256.json"
    if os.path.exists(path):
        return json.load(open(path))
    import kl_eval
    from perseus.examples.gpt2 import ref
    from perseus.impl import config as pconfig
    CF = pconfig.load_configs(BASE)
    SM = [CF.softmax[f"transformer.h.{l}.attn"] for l in range(12)]
    rng = [[math.inf, 0.0] for _ in range(12)]
    orig = ref.softmax_ref

    def tap(S, cfg, kc):
        l = SM.index(cfg)
        d = np.exp((S - (cfg.clip_hi + cfg.clip_lo) / 2) / 2 ** cfg.log2delta2).sum(-1) / kc
        rng[l] = [min(rng[l][0], float(d.min())), max(rng[l][1], float(d.max()))]
        return orig(S, cfg, kc)
    ref.softmax_ref = tap
    pool = np.load(kl_eval.POOL, mmap_mode="r")
    for i in range(256):
        o = 1_000_000 + 3889 * i
        kl_eval.forward_approx(np.array(pool[o:o + 16]), CF)
    ref.softmax_ref = orig
    json.dump(rng, open(path, "w"))
    return rng


def tighten_softmax(s, l, top, tgt_last):
    """Round 0's band keeps the record's low edge and takes its top at `top` x the calibration max (never above the
    record's); the seed is that band's minimax seed and the counts are redesigned."""
    from perseus.impl import smcheb
    from perseus.impl.config import parse_softmax
    lo, hi = smcheb.band_from_seed(s["init_alpha"], s["init_beta"])
    hi = min(hi, top * round0_ranges()[l][1])
    s["init_alpha"], s["init_beta"] = smcheb.cheb_seed(lo, hi)
    cfg = parse_softmax(s)
    ks, tg = smcheb.design(cfg.init_alpha, cfg.init_beta, cfg.log2delta2, tgt_last,
                           tuple(k for k in smcheb.KC_DESIGN if k <= 16) + (16,))
    s["recip_iters_cheb"], s["recip_targets"] = ks, tg


def apply_split(c, path):
    """sm_split.py's per-layer choice: l1 squarings, l2 rounds, round 0's band seed and the redesigned counts. The
    legacy per-round lists (refine_*, per_step_refine_iters, sm_kc_r) follow the new l2 (cut, or the last repeated)."""
    from perseus.impl import smcheb
    for l, b in json.load(open(path)).items():
        s = c["softmax"][f"transformer.h.{l}.attn"]
        l2_old = int(s["refinement_iters"])
        s["n_squarings"], s["refinement_iters"] = b["l1"], b["l2"]
        s["delta1"], s["delta2"] = 2 ** b["l1"], 2 ** b["l2"]
        s["init_alpha"], s["init_beta"] = smcheb.cheb_seed(b["lo"], b["hi"])
        s["recip_iters_cheb"], s["recip_targets"] = b["ks"], b["tg"]
        for key in ("refine_alpha", "refine_beta", "per_step_refine_iters"):
            v = list(s[key])
            s[key] = (v + v[-1:] * b["l2"])[:b["l2"]] if v else []
        if s.get("sm_kc_r") and l2_old:
            C = len(s["sm_kc_r"]) // l2_old
            rows = [s["sm_kc_r"][i * C:(i + 1) * C] for i in range(l2_old)]
            s["sm_kc_r"] = sum((rows + rows[-1:] * b["l2"])[:b["l2"]], [])
        s["note"] = f"split l1={b['l1']} l2={b['l2']} (sm_split.py, {os.path.basename(path)})"


def build(knobs):
    src = next((k.split("=")[1] for k in knobs if k.startswith("from=")), None)
    c = json.load(open(BASE if src is None else f"{REC}/python/perseus/configs/model/approximation/cand_{src}/configs.json"))
    tgt = next((float(k.split("=")[1]) for k in knobs if k.startswith("smtgt=")), 1e-3)
    for kn in knobs:
        k, v = kn.split("=")
        if k == "gelu":
            for g in c["softgelu"].values():
                if len(g["cheb_coeffs"]) - 1 > int(v):
                    refit_gelu(g, int(v))
        elif k.startswith("gelu."):
            refit_gelu(c["softgelu"][f"transformer.h.{k[5:]}.mlp.act"], int(v))
        elif k == "ln":
            for n in c["norm"].values():
                if len(n["cheb_coeffs"]) - 1 > int(v):
                    refit_ln(n, int(v))
        elif k.startswith("ln."):
            refit_ln(c["norm"][k[3:]], int(v))
        elif k == "smtgt":
            if not any(x.startswith("smtop=") for x in knobs):
                for s in c["softmax"].values():
                    retarget_softmax(s, float(v))
        elif k == "smsplit":
            apply_split(c, f"{W}/sm_split_m{v.split(':')[0]}_t{v.split(':')[1]}.json" if v.count(":") == 1 else
                        f"{W}/sm_split_m{v.split(':')[0]}h{v.split(':')[2]}_t{v.split(':')[1]}.json")
        elif k == "smtop":
            for l in range(12):
                tighten_softmax(c["softmax"][f"transformer.h.{l}.attn"], l, float(v), tgt)
        elif k == "from":
            pass                                   # handled above
        elif k == "defer":
            for l in v.split(","):
                c["softmax"][f"transformer.h.{l}.attn"]["defer_norm"] = True
        else:
            raise SystemExit(f"unknown knob {kn}")
    return c


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name, knobs = args[0], args[1:]
    cdir = f"{REC}/python/perseus/configs/model/approximation/cand_{name}"
    os.makedirs(cdir, exist_ok=True)
    cfg_path = f"{cdir}/configs.json"
    json.dump(build(knobs), open(cfg_path, "w"), indent=1)
    t0 = time.time()
    import kl_eval
    from perseus.impl import config as pconfig
    kl_mean, kl_max = kl_eval.Evaluator("owt", 8).kl(pconfig.load_configs(cfg_path))
    bts = "-"
    if "--no-plan" not in sys.argv and math.isfinite(kl_mean):
        cap = f"{W}/caps/{name}"
        r = subprocess.run(["bash", f"{W}/capture_rec.sh", cap,
                            f"configs/model/approximation/cand_{name}/configs.json"],
                           capture_output=True, text=True)
        if r.returncode:
            bts = "capture-failed"
            open(f"{W}/caps/{name}.log", "w").write(r.stdout + r.stderr)
        else:
            out = subprocess.run(["bash", f"{W}/plan_count.sh", cap, f"cand_{name}"] +
                                 (["ilp"] if "--ilp" in sys.argv else []), capture_output=True, text=True)
            bts = out.stdout.strip().split()[-1] if out.returncode == 0 else "plan-failed"
    line = f"{name}\t{' '.join(knobs)}\t{kl_mean:.3e}\t{kl_max:.3e}\t{bts}\t{'ilp' if '--ilp' in sys.argv else 'mincut'}\t{time.time() - t0:.0f}s"
    print(line)
    with open(f"{W}/search.tsv", "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()

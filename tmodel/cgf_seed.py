"""Round 0 of record 4's softmax division (SM_CHEB_GS, centred softmax): how many Chebyshev-scaled Goldschmidt levels
the round needs with the record's linear seed on its band, against a seed that first divides the divisor by a
per-row estimate exp(E) (E = CGF of the row's log-exponentials: mean, or mean + var/2), computed off the critical path.

The divisor of a row is D = sum_j z_j with z the record's own exponential (ref.softmax_ref up to the division: centred
on the reference key, Chebyshev exp, squarings, / kc). w_j = log z_j is the estimator's input (affine in the scores
up to the exp's approximation error, which the measured band absorbs). Bands are fitted on CAL windows, widened by
MARGIN on each side (the record's x3), and checked on HOLD windows (rows outside count). Levels from impl.smcheb.k_for
at the round's own target (recip_targets[0]).

usage (from work/perseus-r4, LN_CHEB=1 HF_HUB_OFFLINE=1): python ../../tmodel/cgf_seed.py [CONFIG_JSON] [N_CAL N_HOLD MARGIN]"""
import json
import math
import os
import sys

os.environ["SM_CHEB_GS"] = "1"
os.environ.setdefault("LN_CHEB", "1")
sys.path.insert(0, os.path.expanduser("~/pesquisa/perseus-speedrun/scripts"))
import numpy as np  # noqa: E402

import kl_eval as K  # noqa: E402
from perseus.examples.gpt2 import ref  # noqa: E402
from perseus.impl import config as pconfig  # noqa: E402
from perseus.impl import smcheb  # noqa: E402
from perseus.impl.poly import eval_chebyshev  # noqa: E402

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json"
n_cal, n_hold, margin = (int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (48, 48, 3.0)
CF = pconfig.load_configs(cfg_path)
layer_of = {id(CF.softmax[f"transformer.h.{l}.attn"]): l for l in range(12)}
REC = []  # (layer, kc, D, mean_w, var_w)
_orig = ref.softmax_ref


def rec_softmax(scores, cfg, kc, head=None):
    """ref.softmax_ref's exponential (up to the division), recorded per head row; returns the reference softmax."""
    s = np.asarray(scores, dtype=np.float64)
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    if cfg.ref_keys:
        idx = np.array([ref.ref_key(r, kc) for r in cfg.ref_keys[:s.shape[-2]]])
        s = s - np.take_along_axis(s, np.broadcast_to(idx[:, None], s.shape[:-1] + (1,)), -1)
    ct = s - mean
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    z = eval_chebyshev(ref._ops, ct, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = z * z
    z = z / kc
    w = np.log(np.maximum(z, 1e-300))
    l = layer_of[id(cfg)]
    for h in range(z.shape[0]):
        REC.append((l, kc, float(z[h].sum()), float(w[h].mean()), float(w[h].var())))
    return _orig(scores, cfg, kc, head)


ref.softmax_ref = rec_softmax


def collect(wins):
    REC.clear()
    for w in wins:
        K.forward_approx(w, CF)
    return np.array(REC)


pool = np.load(K.POOL, mmap_mode="r")
cal = [np.array(pool[o:o + K.T]) for o in range(300_000, 300_000 + 7919 * n_cal, 7919)]
hold = [np.array(pool[o:o + K.T]) for o in range(900_000, 900_000 + 7919 * n_hold, 7919)]
A, B = collect(cal), collect(hold)
np.savez(os.path.expanduser("~/pesquisa/perseus-speedrun/tmodel/cgf_rows.npz"), cal=A, hold=B)
lm = math.log(margin)
print(f"cal {len(cal)} windows ({len(A)} rows), hold {len(hold)} ({len(B)} rows), margin x{margin}")
print(f"{'L':>2} {'record k0':>9} {'rec band':>9} | {'data D':>8} {'k':>2} out | {'D/e^mu':>8} {'k':>2} out | {'D/e^cgf':>8} {'k':>2} out")
tot = {"rec": 0, "data": 0, "mu": 0, "cgf": 0}
for l in range(12):
    c = CF.softmax[f"transformer.h.{l}.attn"]
    tgt = float(c.recip_targets[0])
    k_rec = int(c.recip_iters_cheb[0])
    lo0, hi0 = smcheb.band_from_seed(c.init_alpha, c.init_beta)
    a, b = A[A[:, 0] == l], B[B[:, 0] == l]
    row = [l, k_rec, f"{math.log10(hi0 / lo0):.2f}"]
    for name, est_a, est_b in (("data", 0 * a[:, 2], 0 * b[:, 2]), ("mu", a[:, 3], b[:, 3]),
                               ("cgf", a[:, 3] + a[:, 4] / 2, b[:, 3] + b[:, 4] / 2)):
        r = np.log(a[:, 2]) - est_a
        lo, hi = r.min() - lm, r.max() + lm
        rb = np.log(b[:, 2]) - est_b
        out = int(((rb < lo) | (rb > hi)).sum())
        k = smcheb.k_for(math.exp(lo), math.exp(hi), tgt)
        tot[name] += k
        row += [f"{(hi - lo) / math.log(10):.2f}", k, out]
    tot["rec"] += k_rec
    print(f"{row[0]:>2} {row[1]:>9} {row[2]:>9} | {row[3]:>8} {row[4]:>2} {row[5]:>3} | {row[6]:>8} {row[7]:>2} {row[8]:>3} | "
          f"{row[9]:>8} {row[10]:>2} {row[11]:>3}")
print("round-0 levels per token (12 layers):", tot, " (band columns: log10 of hi/lo)")

if os.environ.get("WRITE_CFG"):
    # round 0 on the data band: the linear minimax seed of [lo, hi] and its level count (the other rounds unchanged)
    j = json.load(open(cfg_path))
    for l in range(12):
        c = CF.softmax[f"transformer.h.{l}.attn"]
        r = np.log(A[A[:, 0] == l][:, 2])
        lo, hi = math.exp(r.min() - lm), math.exp(r.max() + lm)
        a_, b_ = smcheb.cheb_seed(lo, hi)
        s = j["softmax"][f"transformer.h.{l}.attn"]
        s["init_alpha"], s["init_beta"] = a_, b_
        s["recip_iters_cheb"][0] = smcheb.k_for(lo, hi, float(c.recip_targets[0]))
        s["note"] = s.get("note", "") + f" | round 0 on the data band x{margin} (tmodel/cgf_seed.py): [{lo:.4g}, {hi:.4g}]"
    json.dump(j, open(os.environ["WRITE_CFG"], "w"), indent=1)
    print("[write]", os.environ["WRITE_CFG"])

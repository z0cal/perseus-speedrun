"""The OpenWebText pool's 16-token windows for the softmax band calibration and its held-out check, and the per-layer
pre-softmax scores of float64 GPT-2 on them (batched, causal). Never the dev prompts.

Two disjoint halves of the pool (openwebtext_gpt2_2000000.npy, 1763 document starts):
  calib:   every document start (token after an EOS) in [0, 1M) + random offsets in [0, 1M) (seed 20261027);
  holdout: every document start in [1M, 2M) + random offsets in [1M, 2M) (seed 20261028).
No window straddles 1M. The calibration set leaves out the audit's stress windows (offsets in STRESS +- 16 tokens),
so they stay a test; the holdout set keeps them."""
import math
import os

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

L, H, D, T = 12, 12, 768, 16
dh = D // H
POOL = os.path.expanduser("~/pesquisa/perseus-speedrun/work/hf/perseus/pools/openwebtext_gpt2_2000000.npy")
HALF = 1_000_000
STRESS = (1094059, 1149233, 1678699, 257792, 853437)     # subagent/audit/auditoria.md R2 (KL = inf at x4/x6)
_W = None


def weights():
    global _W
    if _W is None:
        _W = {k: v.astype(np.float64) for k, v in load_file(hf_hub_download("openai-community/gpt2", "model.safetensors")).items()}
    return _W


def offsets(kind, n_random):
    """(document-start offsets, random offsets) of `kind` (calib | holdout)."""
    pool = np.load(POOL, mmap_mode="r")
    lo, hi, seed = (0, HALF, 20261027) if kind == "calib" else (HALF, len(pool), 20261028)
    e = np.where(np.asarray(pool) == 50256)[0] + 1
    docs = [int(o) for o in e if lo <= o and o + T <= hi]
    rnd = sorted(int(o) for o in np.random.default_rng(seed).integers(lo, hi - T, n_random))
    if kind == "calib":
        far = lambda o: all(abs(o - s) >= T for s in STRESS)  # noqa: E731
        docs, rnd = [o for o in docs if far(o)], [o for o in rnd if far(o)]
    return docs, rnd


def windows(offs):
    pool = np.load(POOL, mmap_mode="r")
    return np.stack([np.asarray(pool[o:o + T]).astype(np.int64) for o in offs])


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def scores(toks):
    """Per layer the pre-softmax scores (B, H, T, T) of a batch of windows (B, T) (row t: keys 0..t valid)."""
    Wt = weights()
    B, t = toks.shape
    x = Wt["wte.weight"][toks] + Wt["wpe.weight"][:t]
    mask = np.tril(np.ones((t, t), bool))
    out = []
    for l in range(L):
        p = f"h.{l}."
        a = _ln(x, Wt[p + "ln_1.weight"], Wt[p + "ln_1.bias"])
        qkv = a @ Wt[p + "attn.c_attn.weight"] + Wt[p + "attn.c_attn.bias"]
        q, k, v = (z.reshape(B, t, H, dh).transpose(0, 2, 1, 3) for z in np.split(qkv, 3, -1))
        S = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)
        out.append(S)
        Sm = np.where(mask, S, -np.inf)
        P = np.exp(Sm - Sm.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        x = x + (P @ v).transpose(0, 2, 1, 3).reshape(B, t, D) @ Wt[p + "attn.c_proj.weight"] + Wt[p + "attn.c_proj.bias"]
        a = _ln(x, Wt[p + "ln_2.weight"], Wt[p + "ln_2.bias"])
        x = x + _gelu(a @ Wt[p + "mlp.c_fc.weight"] + Wt[p + "mlp.c_fc.bias"]) @ Wt[p + "mlp.c_proj.weight"] \
            + Wt[p + "mlp.c_proj.bias"]
    return out


def divisors(S, mid, l2):
    """Round 0's divisor d = mean over the t + 1 keys of exp((S - mid) / 2^l2), per (B, H, query t)."""
    tri = np.tril(np.ones((T, T), bool))
    return np.where(tri, np.exp((S - mid) / 2 ** l2), 0.0).sum(-1) / np.arange(1, T + 1)

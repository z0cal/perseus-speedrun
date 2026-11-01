"""KL of the Decode-8 approximations (float64, Perseus's own ref primitives) for a modified approximation config.

The model: teacher-forced decode of 16 positions, LayerNorm by `ref.layer_norm_ref`, softmax by `ref.softmax_ref`
(SM_CHEB_GS=1, the record's runtime), GELU as GELU_FOLD runs its Chebyshev series; CutMax left out (it reads the
logits, not the KL). KL(exact float64 GPT-2 || approximate) of the logits at positions 8..15, mean and max.

    python -I kl_eval.py CONFIG_JSON [--windows owt|dev] [--n N]

As a module: `Evaluator(windows).kl(configs)` with a parsed `perseus.impl.config.Configs`.
"""
import argparse
import json
import math
import os
import sys

os.environ["SM_CHEB_GS"] = "1"
import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

from perseus.examples.gpt2 import ref  # noqa: E402
from perseus.impl import config as pconfig  # noqa: E402
from perseus.impl.poly import eval_chebyshev  # noqa: E402

L, H, D, T = 12, 12, 768, 16
dh = D // H
W = {k: v.astype(np.float64) for k, v in load_file(hf_hub_download("openai-community/gpt2", "model.safetensors")).items()}
POOL = os.path.expanduser("~/pesquisa/perseus-speedrun/work/hf/perseus/pools/openwebtext_gpt2_2000000.npy")
DEV = os.path.expanduser("~/pesquisa/perseus-speedrun/work/perseus-rec/speedrun/prompts/dev.json")


def windows(kind, n):
    """`owt`: 16-token windows of the calibration pool away from the record's capture rows (4096..16384);
    `dev`: the speedrun's public dev prompts (validation only)."""
    if kind == "dev":
        return [np.array(p[:T]) for p in json.load(open(DEV))["prompts"][:n]]
    pool = np.load(POOL, mmap_mode="r")
    return [np.array(pool[o:o + T]) for o in range(100_000, 100_000 + 7919 * n, 7919)]


def ln_exact(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def gelu_exact(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def forward_exact(toks):
    t = len(toks)
    x = W["wte.weight"][toks] + W["wpe.weight"][:t]
    mask = np.tril(np.ones((t, t), bool))[None]
    for l in range(L):
        p = f"h.{l}."
        a = ln_exact(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        q, k, v = (z.reshape(t, H, dh).transpose(1, 0, 2)
                   for z in np.split(a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"], 3, -1))
        S = np.where(mask, q @ k.transpose(0, 2, 1) / math.sqrt(dh), -np.inf)
        P = np.exp(S - S.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        x = x + (P @ v).transpose(1, 0, 2).reshape(t, D) @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        a = ln_exact(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        x = x + gelu_exact(a @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]) @ W[p + "mlp.c_proj.weight"] \
            + W[p + "mlp.c_proj.bias"]
    return ln_exact(x, W["ln_f.weight"], W["ln_f.bias"]) @ W["wte.weight"].T


def forward_approx(toks, CF):
    _ops = ref._ops
    SM = [CF.softmax[f"transformer.h.{l}.attn"] for l in range(L)]
    x = W["wte.weight"][toks] + W["wpe.weight"][: len(toks)]
    Ks, Vs, out = [[] for _ in range(L)], [[] for _ in range(L)], []
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
            P = ref.softmax_ref(S, SM[l], t + 1)
            h = h + np.einsum("hk,khd->hd", P, V).reshape(D) @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
            a = ref.layer_norm_ref(h, CF.norm[f"transformer.h.{l}.ln_2"], t, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
            g = CF.gelu[f"transformer.h.{l}.mlp.act"]
            u = a @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]
            u = g.xmax * eval_chebyshev(_ops, u / g.xmax, g.cheb_coeffs, g.cheb_a, g.cheb_b, g.cheb_eps) \
                if g.method == "chebyshev" else ref.gelu_ref(u, g)
            h = h + u @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        out.append(ref.layer_norm_ref(h, CF.norm["transformer.ln_f"], t, W["ln_f.weight"], W["ln_f.bias"])
                   @ W["wte.weight"].T)
    return np.array(out)


def logsoftmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


class Evaluator:
    def __init__(self, kind="owt", n=8):
        self.wins = windows(kind, n)
        self.ref = [logsoftmax(forward_exact(w)) for w in self.wins]

    def kl(self, CF):
        ks = []
        with np.errstate(all="ignore"):
            for w, lp in zip(self.wins, self.ref):
                lq = logsoftmax(forward_approx(w, CF))
                ks.append((np.exp(lp) * (lp - lq)).sum(-1)[8:T])
        k = np.concatenate(ks)
        return (float(np.mean(k)), float(np.max(k))) if np.all(np.isfinite(k)) else (math.inf, math.inf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--windows", default="owt", choices=("owt", "dev"))
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    m, x = Evaluator(a.windows, a.n).kl(pconfig.load_configs(a.config))
    print(f"KL pos 8-15 over {a.n} {a.windows} windows: mean {m:.3e} max {x:.3e}")

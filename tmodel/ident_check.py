"""Float64 check of two exact identities on GPT-2 small (no Perseus code):
  A) centred residual stream: fold C = I - 11^T/d into the outputs of attn.c_proj and mlp.c_proj (weights and
     biases) and centre x0 once; every LayerNorm then equals an RMSNorm (no mean), exactly.
  B) vocab-centred LM head: W_e' = W_e - mean over the vocab rows; logits shift by a per-token constant, so the
     KL and the argmax are unchanged.
Then the same with a noise e added to the residual stream after every sublayer (a stand-in for refresh noise),
to see whether skipping the mean (the noise's mean is no longer removed) costs KL.
usage: python -I ident_check.py MODEL.safetensors"""
import sys

import numpy as np

sys.path.insert(0, "/home/zocal/pesquisa/perseus-speedrun/work/perseus-r4/speedrun/harness")
from gpt2_ref import _gelu, _ln, kl, load_safetensors  # noqa: E402

w = load_safetensors(sys.argv[1])
d, H, L = 768, 12, 12
hd = d // H


def rms(x, g, b, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g + b


def centre_out(W, b):
    return W - W.mean(axis=1, keepdims=True), b - b.mean()


def forward(rows, centred=False, noise=0.0, rng=None):
    x = np.array(rows, dtype=np.float64)
    T = x.shape[0]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    if centred:
        x = x - x.mean(-1, keepdims=True)              # once, at block 0 (the LN1 there computes this mean anyway)
    norm = rms if centred else _ln
    nz = (lambda: noise * rng.standard_normal(x.shape)) if noise else (lambda: 0.0)
    for i in range(L):
        p = f"h.{i}."
        Wo, bo = w[p + "attn.c_proj.weight"], w[p + "attn.c_proj.bias"]
        Wd, bd = w[p + "mlp.c_proj.weight"], w[p + "mlp.c_proj.bias"]
        if centred:
            Wo, bo = centre_out(Wo, bo)
            Wd, bd = centre_out(Wd, bd)
        q, k, v = np.split(norm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"]) @ w[p + "attn.c_attn.weight"]
                           + w[p + "attn.c_attn.bias"], 3, axis=-1)
        q, k, v = (a.reshape(T, H, hd).transpose(1, 0, 2) for a in (q, k, v))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        a = (s / s.sum(-1, keepdims=True)) @ v
        x = x + a.transpose(1, 0, 2).reshape(T, d) @ Wo + bo + nz()
        h = _gelu(norm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"]) @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"])
        x = x + h @ Wd + bd + nz()
    We = w["wte.weight"]
    if centred:
        We = We - We.mean(axis=0, keepdims=True)
    return norm(x, w["ln_f.weight"], w["ln_f.bias"]) @ We.T, x


rng = np.random.default_rng(12345)
ids = rng.integers(0, w["wte.weight"].shape[0], 16)     # public random ids (not the dev prompts)
rows = w["wte.weight"][ids] + w["wpe.weight"][:16]
ref, xr = forward(rows)
cen, xc = forward(rows, centred=True)
shift = (cen - ref).mean(-1, keepdims=True)
print("A+B exact: max |logits' - logits - shift| =", float(np.abs(cen - ref - shift).max()),
      "| max |x mean| centred stream =", float(np.abs(xc.mean(-1)).max()),
      "| argmax equal:", bool((cen.argmax(-1) == ref.argmax(-1)).all()),
      "| max KL =", max(kl(ref[t], cen[t]) for t in range(16)))
for sig in (1e-4, 1e-3, 1e-2):
    ko, kc = [], []
    for rep in range(4):
        o, _ = forward(rows, noise=sig, rng=np.random.default_rng(rep))
        c, _ = forward(rows, centred=True, noise=sig, rng=np.random.default_rng(rep))
        ko.append(np.mean([kl(ref[t], o[t]) for t in range(8, 16)]))
        kc.append(np.mean([kl(ref[t], c[t]) for t in range(8, 16)]))
    print(f"stream noise {sig:g}: mean KL (pos 8-15) LayerNorm {np.mean(ko):.3e}  centred+RMS {np.mean(kc):.3e}")

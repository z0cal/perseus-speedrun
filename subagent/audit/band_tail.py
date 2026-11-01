"""Audit: round-0 Goldschmidt band headroom of the final config on windows the search never saw.

Exact float64 GPT-2 (batched, causal) on 16-token windows; per layer, per position t (kc = t + 1 keys) and head, the
round-0 divisor d = mean_k exp((S_k - mid) / 2^l2) as scripts/headroom_split.py computes it (mid = (clip_hi + clip_lo)
/ 2 of the layer's softmax config, l2 = log2(delta2)). Reports per layer min(d)/lo and hi/max(d) (< 1 = outside the
band, where the Chebyshev-scaled Goldschmidt diverges), and the same for the record's config.

    python band_tail.py docstarts|random N
docstarts: the N first OpenWebText document starts of the calibration pool (token after each EOS), as C4/dev prompts
are document starts; random: N offsets below 1,000,000 (the calibration used 1,000,000 + 3889 i)."""
import json
import math
import os
import sys

import numpy as np
from safetensors.numpy import load_file
from huggingface_hub import hf_hub_download

L, H, D, T = 12, 12, 768, 16
dh = D // H
Wt = {k: v.astype(np.float64) for k, v in load_file(hf_hub_download("openai-community/gpt2", "model.safetensors")).items()}
POOL = os.path.expanduser("~/pesquisa/perseus-speedrun/work/hf/perseus/pools/openwebtext_gpt2_2000000.npy")
NEW = os.path.expanduser("~/pesquisa/perseus-speedrun/configs/cand_dfr13_cm/configs.json")
SPLIT = os.path.expanduser("~/pesquisa/perseus-speedrun/work/sm_split_m4h6_t0.01.json")
REC = os.path.expanduser("~/pesquisa/perseus-speedrun/work/perseus-rec/python/perseus/configs/model/approximation/"
                         "gpt2_base_n32_kc16_cm/configs.json")


def ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def scores(toks):
    """Per layer the pre-softmax scores (B, H, T, T) of a batch of windows (B, T)."""
    B, t = toks.shape
    x = Wt["wte.weight"][toks] + Wt["wpe.weight"][:t]
    mask = np.tril(np.ones((t, t), bool))
    out = []
    for l in range(L):
        p = f"h.{l}."
        a = ln(x, Wt[p + "ln_1.weight"], Wt[p + "ln_1.bias"])
        qkv = a @ Wt[p + "attn.c_attn.weight"] + Wt[p + "attn.c_attn.bias"]
        q, k, v = (z.reshape(B, t, H, dh).transpose(0, 2, 1, 3) for z in np.split(qkv, 3, -1))
        S = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)
        out.append(S)
        Sm = np.where(mask, S, -np.inf)
        P = np.exp(Sm - Sm.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        x = x + (P @ v).transpose(0, 2, 1, 3).reshape(B, t, D) @ Wt[p + "attn.c_proj.weight"] + Wt[p + "attn.c_proj.bias"]
        a = ln(x, Wt[p + "ln_2.weight"], Wt[p + "ln_2.bias"])
        x = x + gelu(a @ Wt[p + "mlp.c_fc.weight"] + Wt[p + "mlp.c_fc.bias"]) @ Wt[p + "mlp.c_proj.weight"] \
            + Wt[p + "mlp.c_proj.bias"]
    return out


def main():
    kind, n = sys.argv[1], int(sys.argv[2])
    pool = np.load(POOL, mmap_mode="r")
    if kind == "docstarts":
        e = np.where(np.asarray(pool) == 50256)[0]
        offs = [int(o) + 1 for o in e[:n] if o + 1 + T <= len(pool)]
    else:
        rng = np.random.default_rng(12345)
        offs = sorted(int(o) for o in rng.integers(0, 1_000_000 - T, n))
    wins = np.stack([np.asarray(pool[o:o + T]).astype(np.int64) for o in offs])
    new = json.load(open(NEW))["softmax"]
    rec = json.load(open(REC))["softmax"]
    sp = {int(k): v for k, v in json.load(open(SPLIT)).items()}
    mn = {c: [math.inf] * L for c in ("new", "rec")}
    mx = {c: [0.0] * L for c in ("new", "rec")}
    worst = {c: [None] * L for c in ("new", "rec")}
    bad = np.zeros(len(wins), bool)
    tri = np.tril(np.ones((T, T), bool))
    kc = np.arange(1, T + 1)[:, None]
    for i in range(0, len(wins), 64):
        Ss = scores(wins[i:i + 64])
        for l in range(L):
            for c, cfg in (("new", new), ("rec", rec)):
                s = cfg[f"transformer.h.{l}.attn"]
                l2 = sp[l]["l2"] if c == "new" else int(round(math.log2(s["delta2"])))
                mid = (s["clip_hi"] + s["clip_lo"]) / 2
                d = np.where(tri, np.exp((Ss[l] - mid) / 2 ** l2), 0.0).sum(-1) / kc[:, 0]  # (B, H, T)
                lo_i, hi_i = np.unravel_index(d.argmin(), d.shape), np.unravel_index(d.argmax(), d.shape)
                if c == "new":
                    out_b = ((d > sp[l]["hi"]) | (d < sp[l]["lo"])).any(axis=(1, 2))
                    bad[i:i + len(out_b)] |= out_b
                if d.min() < mn[c][l]:
                    mn[c][l] = float(d.min())
                if d.max() > mx[c][l]:
                    mx[c][l] = float(d.max())
                    worst[c][l] = (offs[i + hi_i[0]], int(hi_i[1]), int(hi_i[2]))
    print(f"{kind}: {len(wins)} windows x 16 positions; windows with a divisor outside a new band: {int(bad.sum())} "
          f"({100 * bad.mean():.2f} %): offsets {[offs[j] for j in np.where(bad)[0][:12]]}")
    for l in range(L):
        s = sp[l]
        print(f"L{l:2d} new l2={s['l2']} band [{s['lo']:.4g}, {s['hi']:.4g}] seen [{mn['new'][l]:.4g}, {mx['new'][l]:.4g}] "
              f"low headroom {mn['new'][l] / s['lo']:.2f} high headroom {s['hi'] / mx['new'][l]:.2f}  (argmax at {worst['new'][l]})")
    print("record (band = its note's head sums x3 when parseable):")
    import re
    for l in range(L):
        note = rec[f"transformer.h.{l}.attn"]["note"]
        m = re.search(r"head sums \[([\d.e-]+), ([\d.e-]+)\] x3", note)
        if m:
            lo, hi = float(m.group(1)) / 3, float(m.group(2)) * 3
            print(f"L{l:2d} rec band~[{lo:.4g}, {hi:.4g}] seen [{mn['rec'][l]:.4g}, {mx['rec'][l]:.4g}] "
                  f"low {mn['rec'][l] / lo:.2f} high {hi / mx['rec'][l]:.2f}")


if __name__ == "__main__":
    main()

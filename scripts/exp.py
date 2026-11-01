"""CGF-softmax (arXiv 2602.01621) against the Perseus speedrun's GPT-2 setting, in float64 plaintext.

Part 1: end-to-end KL(p_ref || p_variant) of the final logits with every attention softmax replaced by
        exp(x - K(1) - ln n), K(1) truncated at order 2/3/4 (no fine-tuning: the speedrun fixes the weights).
        Decode-8 = positions 8..15 (causal model, teacher-forced => same as the prefix forward); Prefill-128 = pos 127.
Part 2: the softmax denominator s = sum_j exp(x_j) of every attention row, and how well each seed y0 ~ 1/s
        flattens it: R = max(s y0) / min(s y0) over rows (per layer). With the best constant scaling the
        Goldschmidt error starts at e0 = (R-1)/(R+1) and after k plain iterations is e0^(2^k).
usage: python -I exp.py PROMPTS_JSON
"""
import json
import math
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

W = load_file(hf_hub_download("openai-community/gpt2", "model.safetensors"))
W = {k: v.astype(np.float64) for k, v in W.items()}
L, H, D = 12, 12, 768
dh = D // H


def ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def sm_exact(S, mask):
    S = np.where(mask, S, -np.inf)
    S = S - S.max(-1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(-1, keepdims=True)


def sm_cgf(order):
    def f(S, mask):
        n = mask.sum(-1, keepdims=True)
        mu = np.where(mask, S, 0).sum(-1, keepdims=True) / n
        c = np.where(mask, S - mu, 0)
        m2 = (c ** 2).sum(-1, keepdims=True) / n
        K = mu + m2 / 2
        if order >= 3:
            K = K + (c ** 3).sum(-1, keepdims=True) / n / 6
        if order >= 4:
            m4 = (c ** 4).sum(-1, keepdims=True) / n
            K = K + (m4 - 3 * m2 ** 2) / 24
        return np.where(mask, np.exp(S - K - np.log(n)), 0)
    return f


def forward(toks, softmax, rows=None):
    T = len(toks)
    x = W["wte.weight"][toks] + W["wpe.weight"][:T]
    mask = np.tril(np.ones((T, T), bool))[None]
    for l in range(L):
        p = f"h.{l}."
        a = ln(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = (t.reshape(T, H, dh).transpose(1, 0, 2) for t in np.split(qkv, 3, -1))
        S = q @ k.transpose(0, 2, 1) / math.sqrt(dh)
        if rows is not None:
            rows.append(S)
        P = softmax(S, mask)
        o = (P @ v).transpose(1, 0, 2).reshape(T, D)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        a = ln(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        x = x + gelu(a @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]) @ W[p + "mlp.c_proj.weight"] \
            + W[p + "mlp.c_proj.bias"]
    x = ln(x, W["ln_f.weight"], W["ln_f.bias"])
    return x @ W["wte.weight"].T


def logsoftmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def kl(ref, alt):
    lp, lq = logsoftmax(ref), logsoftmax(alt)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def lr_after_poly(lD, deg):
    """log R left by the best relative seed of degree `deg` for 1/s on a range of log-width lD:
    min max |1 - s p(s)| = 1/T_{deg+1}((D+1)/(D-1)) = 1/cosh(2 (deg+1) artanh(D^-1/2))."""
    e = 1 / math.cosh(min(2 * (deg + 1) * math.atanh(math.exp(-lD / 2)), 700))
    return math.log1p(e) - math.log1p(-e) if e < 1 else lD


def iters(lR, tau=1e-3, cheb=False):
    """Iterations to |e| < tau from a residual range R = exp(lR): plain Goldschmidt e0^(2^k), e0 = (R-1)/(R+1),
    -ln e0 = 2 artanh(1/R); or Chebyshev-scaled steps (SM_CHEB_GS), optimal degree 2^k:
    1/cosh(2^k * 2 artanh(R^-1/2)) < tau. Logs throughout: R overflows a float."""
    if cheb:
        lg = math.log(2 * math.atanh(math.exp(-lR / 2))) if lR < 600 else math.log(2) - lR / 2
        need = math.log(math.acosh(1 / tau))
    else:
        lg = math.log(2 * math.atanh(math.exp(-lR))) if lR < 600 else math.log(2) - lR
        need = math.log(-math.log(tau))
    return max(0, math.ceil((need - lg) / math.log(2)))


def main():
    prompts = json.load(open(sys.argv[1]))["prompts"]
    print(f"{len(prompts)} prompts x {len(prompts[0])} tokens\n")

    # ---- Part 1
    variants = {} if len(sys.argv) > 2 else {"cgf2": sm_cgf(2)}
    res = {k: {"dec": [], "pf": []} for k in variants}
    srows = [[] for _ in range(L)]       # per layer: list of (s, n, mu, m2, m3) over rows
    for toks in prompts:
        rows = []
        ref = forward(np.array(toks), sm_exact, rows)
        for l, S in enumerate(rows):
            T = S.shape[-1]
            for i in range(T):
                x = S[:, i, : i + 1]
                n = i + 1
                mu = x.mean(-1)
                c = x - mu[:, None]
                mx = x.max(-1)
                srows[l].append(np.stack([mx + np.log(np.exp(x - mx[:, None]).sum(-1)), np.full(H, n), mu, (c ** 2).mean(-1),
                                          (c ** 3).mean(-1)], 1))
        for k, f in variants.items():  # noqa
            alt = forward(np.array(toks), f)
            d = kl(ref, alt)
            res[k]["dec"].append(d[8:16])
            res[k]["pf"].append(d[127])
    if variants:
        print("Part 1  KL(ref || CGF softmax), no fine-tuning   [bars: decode < 1e-3, prefill < 1e-2]")
        print(f"{'variant':8} {'decode mean':>12} {'decode max':>12} {'prefill mean':>13} {'prefill max':>12}")
    for k in variants:
        dd, pp = np.concatenate(res[k]["dec"]), np.array(res[k]["pf"])
        print(f"{k:8} {dd.mean():12.3e} {dd.max():12.3e} {pp.mean():13.3e} {pp.max():12.3e}")

    # ---- Part 2
    print("\nPart 2  residual range R = max(s*y0)/min(s*y0) per layer, rows of positions 0..127 "
          "(decode-16 rows in brackets)")
    print("        (plain/Chebyshev-scaled Goldschmidt iterations to |e| < 1e-3); seed depth: poly_d = ceil(log2(d+1)),"
          " cgf = sigma^2 + one more exp")
    hdr = f"{'L':>2} {'D=no seed':>16} {'poly1':>14} {'poly3':>14} {'poly7':>14} {'cgf2':>16} {'cgf3':>16} {'cgf2 diverge':>12}"
    print(hdr)
    for l in range(L):
        a = np.concatenate(srows[l])
        s, n, mu, m2, m3 = a.T
        out = []
        for sel in (slice(None), n <= 16):
            ls = s[sel]                                   # log s
            lD = ls.max() - ls.min()
            l2 = ls - np.log(n[sel]) - mu[sel] - m2[sel] / 2   # log(s / s_cgf2)
            l3 = l2 - m3[sel] / 6
            out.append(dict(D=lD, p1=lr_after_poly(lD, 1), p3=lr_after_poly(lD, 3), p7=lr_after_poly(lD, 7),
                            c2=l2.max() - l2.min(), c3=l3.max() - l3.min(),
                            div=(l2 - np.median(l2) >= math.log(2)).mean()))
        f = lambda key: (f"{math.exp(min(out[0][key], 700)):.1e}({iters(out[0][key])}/{iters(out[0][key], cheb=True)})"
                         f"[{iters(out[1][key])}/{iters(out[1][key], cheb=True)}]")
        print(f"{l:>2} {f('D'):>16} {f('p1'):>14} {f('p3'):>14} {f('p7'):>14} {f('c2'):>16} {f('c3'):>16} "
              f"{out[0]['div']:>12.1%}")


if __name__ == "__main__":
    main()

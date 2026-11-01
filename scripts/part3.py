"""Round 0 of the Decode-8 record's softmax division (SM_CHEB_GS): the divisor is s = sum_j exp(x_j / delta2) over the
row's n <= 16 keys. Chebyshev-scaled Goldschmidt iterations needed to reach the round's own target (recip_targets[0])
after a linear seed, for three bands, each widened by MARGIN on both ends as the record's (its "x3"):
  global : one band over every row (what the record does; should reproduce its recip_iters_cheb[0]);
  per-n  : one band per key count n (public: the decode position), the worst n counted;
  cgf2   : s divided by n exp(mu + sigma^2/2) first (costs sigma^2 and one more exp; not counted here).
usage: python -I part3.py PROMPTS_JSON CONFIG_JSON [MARGIN]"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exp  # noqa: E402

cfg = json.load(open(sys.argv[2]))["softmax"]
margin = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
lm = 2 * math.log(margin)
C = [cfg[f"transformer.h.{l}.attn"] for l in range(exp.L)]
prompts = json.load(open(sys.argv[1]))["prompts"]
st = [[] for _ in range(exp.L)]
for toks in prompts:
    rows = []
    exp.forward(np.array(toks[:16]), exp.sm_exact, rows)
    for l, S in enumerate(rows):
        S = S / C[l]["delta2"]
        for i in range(16):
            x = S[:, i, : i + 1]
            mx = x.max(-1)
            mu = x.mean(-1)
            c = x - mu[:, None]
            st[l].append(np.stack([mx + np.log(np.exp(x - mx[:, None]).sum(-1)), np.full(exp.H, i + 1), mu,
                                   (c ** 2).mean(-1)], 1))


def k_after_linear(lD, tau):
    return exp.iters(exp.lr_after_poly(lD + lm, 1), tau=tau, cheb=True) + 1   # + the seed's own step


print(f"{len(prompts)} prompts, margin x{margin:g}; Chebyshev Goldschmidt iterations of round 0 (seed step included)")
print(f"{'L':>2} {'delta2':>6} {'target':>6} {'record':>6} {'global':>6} {'per-n':>6} {'cgf2':>6}   band log-widths g/n/cgf")
for l in range(exp.L):
    ls, n, mu, m2 = np.concatenate(st[l]).T
    tau = float(C[l]["recip_targets"][0])
    lg = np.ptp(ls)
    ln_ = max(np.ptp(ls[n == k]) for k in range(1, 17))
    lc = np.ptp(ls - np.log(n) - mu - m2 / 2)
    print(f"{l:>2} {C[l]['delta2']:>6g} {tau:>6g} {C[l]['recip_iters_cheb'][0]:>6} {k_after_linear(lg, tau):>6} "
          f"{k_after_linear(ln_, tau):>6} {k_after_linear(lc, tau):>6}   {lg:.2f}/{ln_:.2f}/{lc:.2f}")

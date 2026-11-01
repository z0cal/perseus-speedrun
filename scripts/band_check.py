"""Round-0 Goldschmidt band headroom of a Decode-8 config on held-out windows (no dev prompts).

The band is read from the config itself (smcheb.band_from_seed of init_alpha / init_beta), l2 = refinement_iters,
mid = (clip_hi + clip_lo) / 2; the divisor is owt_windows.divisors on float64 GPT-2 scores. Per layer: the band, the
divisors seen, low headroom min(d) / lo and high headroom hi / max(d) (< 1 = outside: above the top the scaled
Goldschmidt diverges), and the windows with a divisor outside any band.

    python -I band_check.py CONFIG_JSON [CONFIG_JSON ...] [holdout|calib] [N_RANDOM]   (defaults holdout, 16384)
Several configs share one pass over the windows."""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import owt_windows as ow  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/pesquisa/perseus-speedrun/work/perseus-rec/python"))
from perseus.impl import smcheb  # noqa: E402


def bands(path):
    sm = json.load(open(path))["softmax"]
    cfg = []
    for l in range(12):
        s = sm[f"transformer.h.{l}.attn"]
        lo, hi = smcheb.band_from_seed(s["init_alpha"], s["init_beta"])
        cfg.append((int(s["refinement_iters"]), (s["clip_hi"] + s["clip_lo"]) / 2, lo, hi))
    return cfg


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = [a for a in args if a.endswith(".json")]
    rest = [a for a in args if not a.endswith(".json")]
    kind = rest[0] if rest else "holdout"
    n = int(rest[1]) if len(rest) > 1 else 16384
    cfgs = {p: bands(p) for p in paths}
    docs, rnd = ow.offsets(kind, n)
    offs = docs + rnd
    mn = {p: [math.inf] * 12 for p in paths}
    mx = {p: [0.0] * 12 for p in paths}
    arg = {p: [None] * 12 for p in paths}
    bad = {p: np.zeros(len(offs), bool) for p in paths}
    t0 = time.time()
    for i in range(0, len(offs), 64):
        Ss = ow.scores(ow.windows(offs[i:i + 64]))
        for p, cfg in cfgs.items():
            for l, (l2, mid, lo, hi) in enumerate(cfg):
                d = ow.divisors(Ss[l], mid, l2)
                bad[p][i:i + len(d)] |= ((d > hi) | (d < lo)).any(axis=(1, 2))
                mn[p][l] = min(mn[p][l], float(d.min()))
                if d.max() > mx[p][l]:
                    mx[p][l], arg[p][l] = float(d.max()), offs[i + int(np.unravel_index(d.argmax(), d.shape)[0])]
        if i % 4096 == 0:
            print(f"  {kind} {i}/{len(offs)} {time.time() - t0:.0f} s", flush=True)
    for p, cfg in cfgs.items():
        b = bad[p]
        print(f"\n{p}\n{kind}: {len(offs)} windows ({len(docs)} document starts) x 16 positions; outside a band: "
              f"{int(b.sum())} {[offs[j] for j in np.where(b)[0][:12]]}")
        for l, (l2, mid, lo, hi) in enumerate(cfg):
            print(f"L{l:2d} l2={l2} band [{lo:.4g}, {hi:.4g}] seen [{mn[p][l]:.4g}, {mx[p][l]:.4g}] low headroom "
                  f"{mn[p][l] / lo:.2f} high headroom {hi / mx[p][l]:.2f} (max at {arg[p][l]})")
        print(f"min headroom: low {min(mn[p][l] / c[2] for l, c in enumerate(cfg)):.2f}, "
              f"high {min(c[3] / mx[p][l] for l, c in enumerate(cfg)):.2f}", flush=True)


if __name__ == "__main__":
    main()

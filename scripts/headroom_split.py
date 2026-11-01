"""Headroom of a split's round-0 bands (sm_split_*.json) on held-out windows (dev 32 + owt 48): per layer, how far
inside [lo, hi] the held-out divisors stay (min(d) / lo and hi / max(d)); < 1 means a held-out row falls outside.
usage: python -I headroom_split.py SPLIT_JSON"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kl_eval
from perseus.examples.gpt2 import ref
from perseus.impl import config as pconfig
W = os.path.expanduser("~/pesquisa/perseus-speedrun/work")
CF = pconfig.load_configs(f"{W}/perseus-rec/python/perseus/configs/model/approximation/gpt2_base_n32_kc16/configs.json")
SM = [CF.softmax[f"transformer.h.{l}.attn"] for l in range(12)]
sp = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
mn, mx = [math.inf] * 12, [0.0] * 12
orig = ref.softmax_ref
def tap(S, cfg, kc):
    l = SM.index(cfg)
    d = np.exp((S - (cfg.clip_hi + cfg.clip_lo) / 2) / 2 ** sp[l]["l2"]).sum(-1) / kc
    mn[l], mx[l] = min(mn[l], float(d.min())), max(mx[l], float(d.max()))
    return orig(S, cfg, kc)
ref.softmax_ref = tap
for w in kl_eval.windows("dev", 32) + kl_eval.windows("owt", 48):
    kl_eval.forward_approx(w, CF)
lows = [mn[l] / sp[l]["lo"] for l in range(12)]
highs = [sp[l]["hi"] / mx[l] for l in range(12)]
print("low headroom :", " ".join(f"{x:.1f}" for x in lows), "| worst", f"{min(lows):.2f}")
print("high headroom:", " ".join(f"{x:.1f}" for x in highs), "| worst", f"{min(highs):.2f}")

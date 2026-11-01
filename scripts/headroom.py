"""Round-0 divisor maxima on held-out windows (dev 32 + owt 48) against the calibration maxima (round0_ranges256.json):
per layer, held-out max / calibration max. A candidate with top = M x calibration max keeps M / ratio of headroom."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kl_eval
from perseus.examples.gpt2 import ref
from perseus.impl import config as pconfig
W = os.path.expanduser("~/pesquisa/perseus-speedrun/work")
CF = pconfig.load_configs(f"{W}/perseus-rec/python/perseus/configs/model/approximation/gpt2_base_n32_kc16/configs.json")
SM = [CF.softmax[f"transformer.h.{l}.attn"] for l in range(12)]
cal = json.load(open(f"{W}/round0_ranges256.json"))
mx = [0.0] * 12
orig = ref.softmax_ref
def tap(S, cfg, kc):
    l = SM.index(cfg)
    mx[l] = max(mx[l], float((np.exp((S - (cfg.clip_hi + cfg.clip_lo) / 2) / 2 ** cfg.log2delta2).sum(-1) / kc).max()))
    return orig(S, cfg, kc)
ref.softmax_ref = tap
for w in kl_eval.windows("dev", 32) + kl_eval.windows("owt", 48):
    kl_eval.forward_approx(w, CF)
r = [mx[l] / cal[l][1] for l in range(12)]
print("held-out max / calibration max per layer:", " ".join(f"{x:.2f}" for x in r), "| worst", f"{max(r):.2f}")
print("headroom at top x3:", f"{3 / max(r):.2f}x", " at x5:", f"{5 / max(r):.2f}x")

"""Sanity: (1) build_cfg.build() in memory (no writes) from the old and new split files vs the shipped configs, byte
for byte as build_cfg would dump them; (2) structural checks of cand_rc3's softmax: l1/l2 fields consistent, recip
counts = 1 + l2, exp domain = the clip half-width, band (from the seed) = the split file's band, no l2 = 0, structural
top >= 1.05 exp((clip_hi - mid)/2^l2), padded-head divisor exp(-mid/2^l2) inside, and the record's band for reference."""
import json, math, os, sys
D = os.path.expanduser("~/pesquisa/perseus-speedrun")
sys.path.insert(0, f"{D}/scripts")
import build_cfg  # noqa: E402
from perseus.impl import smcheb  # noqa: E402  (record worktree's, via search's sys.path)
W = f"{D}/work"
for name, split in (("cand_dfr13_cm", "sm_split_m4h6_t0.01.json"), ("cand_rc3", "sm_split_rc3_t0.01.json")):
    c = build_cfg.build(f"{W}/{split}")
    s = json.dumps(c, indent=1)
    f = open(f"{D}/configs/{name}/configs.json").read()
    print(f"{name} from {split}: rebuilt == shipped bytes: {s == f}  (len {len(s)} vs {len(f)})")
c = json.load(open(f"{D}/configs/cand_rc3/configs.json"))
r = json.load(open(f"{W}/perseus-rec/python/perseus/configs/model/approximation/gpt2_base_n32_kc16_cm/configs.json"))
sp = json.load(open(f"{W}/sm_split_rc3_t0.01.json"))
for l in range(12):
    s = c["softmax"][f"transformer.h.{l}.attn"]
    q = r["softmax"][f"transformer.h.{l}.attn"]
    l1, l2 = int(s["n_squarings"]), int(s["refinement_iters"])
    ok_f = (2 ** l1 == s["delta1"]) and (2 ** l2 == s["delta2"]) and len(s["recip_iters_cheb"]) == 1 + l2 and \
        len(s["recip_targets"]) == 1 + l2 and len(s["refine_alpha"]) == l2 == len(s["refine_beta"]) == len(s["per_step_refine_iters"])
    mid = (s["clip_hi"] + s["clip_lo"]) / 2
    sf = 2.0 ** (-l1 - l2)
    dom = (s["cheb_a"] / sf, s["cheb_b"] / sf)
    lo, hi = smcheb.band_from_seed(s["init_alpha"], s["init_beta"])
    b = sp[str(l)]
    top = 1.05 * math.exp((s["clip_hi"] - mid) / 2 ** l2)
    pad = math.exp(-mid / 2 ** l2)
    rlo, rhi = smcheb.band_from_seed(q["init_alpha"], q["init_beta"])
    ql1, ql2 = int(q["n_squarings"]), int(q["refinement_iters"])
    qsf = 2.0 ** (-ql1 - ql2)
    ks, tg = smcheb.design(s["init_alpha"], s["init_beta"], l2, float(s["recip_targets"][-1]),
                           tuple(k for k in smcheb.KC_DESIGN if k <= 16) + (16,))
    print(f"L{l:2d} l1={l1} l2={l2} fields_ok={ok_f} clip=[{s['clip_lo']:.3f},{s['clip_hi']:.3f}] (rec same: "
          f"{(s['clip_lo'], s['clip_hi']) == (q['clip_lo'], q['clip_hi'])}) exp-dom={dom[0]:.3f},{dom[1]:.3f} rec-dom="
          f"{q['cheb_a'] / qsf:.3f},{q['cheb_b'] / qsf:.3f} | band [{lo:.4g},{hi:.4g}] split-file [{b['lo']:.4g},{b['hi']:.4g}] "
          f"top_ok={hi >= top * (1 - 1e-9)} pad {pad:.3g} in={lo <= pad <= hi} | rec l2={ql2} band [{rlo:.4g},{rhi:.4g}] | "
          f"counts {s['recip_iters_cheb']} redesign {ks} same={list(ks) == list(s['recip_iters_cheb'])} tg {s['recip_targets']}")

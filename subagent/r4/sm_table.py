"""Per-layer softmax structure of record 3 (kc16_cm), record 4 (kc16r_cm) and cand_rc3: l1 = log2 delta1 squarings,
l2 = log2 delta2 normalise-and-square rounds, recip_iters_cheb (THOR rounds' Chebyshev counts), refine, ref_keys."""
import json, math, sys
A = "/home/zocal/pesquisa/perseus-speedrun/work/perseus-r4/python/perseus/configs/model/approximation"
cfgs = {"rec3": f"{A}/gpt2_base_n32_kc16_cm/configs.json", "rec4": f"{A}/gpt2_base_n32_kc16r_cm/configs.json",
        "rc3": "/home/zocal/pesquisa/perseus-speedrun/configs/cand_rc3/configs.json"}
C = {k: json.load(open(v))["softmax"] for k, v in cfgs.items()}
keys = ["delta1", "delta2", "n_squarings", "recip_iters_cheb", "refinement_iters", "per_step_refine_iters", "gs_iters_first", "gs_iters_scaled", "recip_targets"]
tot = {k: [0, 0] for k in C}
for l in range(12):
    print(f"L{l}")
    for k, c in C.items():
        s = c[f"transformer.h.{l}.attn"]
        l1, l2 = round(math.log2(s["delta1"])), round(math.log2(s["delta2"]))
        rounds = len(s["recip_iters_cheb"]) if isinstance(s["recip_iters_cheb"], list) else s["recip_iters_cheb"]
        lv = l1 + sum(s["recip_iters_cheb"]) + l2
        tot[k][0] += rounds; tot[k][1] += lv
        print(f"  {k:4s} l1={l1} l2={l2} nsq={s.get('n_squarings')} cheb={s['recip_iters_cheb']} refine={s.get('refinement_iters')} "
              f"ref_keys={'Y' if s.get('ref_keys') else '-'} defer={s.get('defer_norm', False)} poly_deg={len(s.get('poly_coeffs') or []) - 1}")
print({k: {"recip rounds": v[0], "l1+sum(cheb)+l2": v[1]} for k, v in tot.items()})

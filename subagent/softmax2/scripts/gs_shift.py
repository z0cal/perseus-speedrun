"""Round-0 vs last-round Goldschmidt counts (delta2 = 2 layers): for each mid target m, k0 = k_for(band0, m),
k1 = max over kc 1..16 of k_for(refine_band(kc, m), tgt). Record round-1 stretch slack from the plan (SB#2 landing)."""
import json, os
import numpy as np
import sm2
from perseus.impl import smcheb
UP = sm2.UP
slack1 = {}
for b in range(12):
    p = json.load(open(f"{UP}/bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm/block_{b}_placement.json"))
    g = json.load(open(f"{UP}/graphs/gpt2_decode_python_n32_kc16rm/block_{b}/graph.json"))["nodes"]
    L = p["summary"]["final_named_levels"]; ss = p["sparse_slots"]
    sb = [n["output"] for n in g if ss.get(n["output"]) == 512]
    slack1[b] = [(L[v] - 26) // 2 for v in sb]
for Lr in range(12):
    j, cfg = sm2.layer(Lr)
    lo0, hi0 = smcheb.band_from_seed(cfg.init_alpha, cfg.init_beta)
    rec = [int(x) for x in cfg.recip_iters_cheb]
    if len(rec) != 2:
        print(f"camada {Lr}: delta2 {2**cfg.log2delta2}, recorde {rec}, folga de pouso dos SparseB (niveis) {slack1[Lr]}"); continue
    opts = []
    for m in (0.005, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8):
        k0 = smcheb.k_for(lo0, hi0, m)
        for tgt in (1e-3, 1e-4):
            k1 = max(smcheb.k_for(*smcheb.refine_band(kc, m), tgt) for kc in range(1, 17))
            r = smcheb.resid(*smcheb.refine_band(16, m), k1)
            opts.append((k0, k1, m, tgt, r))
    best = {}
    for k0, k1, m, tgt, r in opts:
        key = (k0, k1)
        if key not in best or r < best[key][2]:
            best[key] = (m, tgt, r)
    s = "  ".join(f"[{k0},{k1}] m={v[0]} res={v[2]:.0e}" for (k0, k1), v in sorted(best.items()) if k0 <= rec[0] and k1 <= rec[1] + 3)
    print(f"camada {Lr}: recorde {rec} (res {smcheb.resid(*smcheb.refine_band(16, float(cfg.recip_targets[0])), rec[1]):.0e}); "
          f"folga SparseB (pousos-26)/2 = {slack1[Lr]} | {s}")

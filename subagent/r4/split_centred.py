"""r4: does another delta2 (our l1/l2 split idea) make a centred layer shallower? Record 4's own rule
(fit_gpt2_decode16_ref.fit_centred): delta1 = next_pow2(W / (5 delta2)), round-0 band [exp(-U/delta2)/16, 1.05],
smcheb.design counts, depth() = 4 + nsq + k0 + 2 + sum(k + 3). delta2 below the record's breaks tau = 4.6 (precision)."""
import json, math, sys
sys.path.insert(0, "/home/zocal/pesquisa/perseus-speedrun/work/perseus-r4/python")
from perseus.impl import smcheb
C = json.load(open("/home/zocal/pesquisa/perseus-speedrun/work/perseus-r4/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json"))["softmax"]
np2 = lambda x: 2 ** math.ceil(math.log2(x))
for l in range(12):
    s = C[f"transformer.h.{l}.attn"]
    if not s.get("ref_keys"):
        continue
    U, lo = s["clip_hi"], s["clip_lo"]; W = U - lo; d2r = s["delta2"]
    row = []
    for d2 in (d2r // 2, d2r, 2 * d2r, 4 * d2r):
        if d2 < 1: continue
        d1 = max(np2(W / (5.0 * d2)), 1)
        ia, ib = smcheb.cheb_seed(math.exp(-U / d2) / 16, 1.05)
        R = int(round(math.log2(d2)))
        try:
            ks, _ = smcheb.design(ia, ib, R, tgt_last=1e-3, kcs=tuple(range(1, 17)))
        except Exception as e:
            row.append(f"d2={d2}: {type(e).__name__}"); continue
        dep = 4 + int(round(math.log2(d1))) + ks[0] + 2 + sum(k + 3 for k in ks[1:])
        row.append(f"d2={d2}{'(rec)' if d2 == d2r else '(<tau)' if d2 < d2r else ''} d1={d1} ks={list(ks)} depth={dep}")
    print(f"L{l}: " + " | ".join(row))

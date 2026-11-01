"""Centred layers rebuilt (h3.build) at other upper margins: depth, counts, round-0 floor, exp-refresh noise proxy
delta2 e^(U/delta2), and the tolerance above U* before the band top breaks (U + 0.74 delta2) in units of U*."""
import math, re
import sm2, h3
for L in range(12):
    j, cfg = sm2.layer(L)
    if not cfg.ref_keys: continue
    m = re.search(r"reach ([\d.]+) .*u_lo (-?[\d.]+)", j["note"])
    us, ul = float(m.group(1)), float(m.group(2))
    row = []
    for mg in (1.25, 1.15, 1.10, 1.0):
        c = h3.build(j, us, ul, margin=mg)
        d2 = c["delta2"]
        row.append(f"m{mg}: d1 {c['delta1']} GS {c['recip_iters_cheb']} prof {h3.depth(c)} ruido {d2 * math.exp(c['clip_hi'] / d2):.0f} "
                   f"tol {(c['clip_hi'] + 0.74 * d2) / us:.2f}U*")
    print(f"camada {L}: U* {us} u_lo {ul} | " + " | ".join(row))

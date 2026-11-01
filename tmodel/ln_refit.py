"""Refit the LayerNorm inverse-sqrt seeds (LN_CHEB) at a looser relative-error target on the record's own bands
(cheb_lo/cheb_hi kept: refit_norm_cheb.py's fit_site, no data pass). usage: python ln_refit.py IN.json OUT.json TARGET"""
import json, math, sys
sys.path.insert(0, "scripts/utils")
from refit_norm_cheb import fit_site  # noqa: E402
from perseus.impl.config import parse_norm  # noqa: E402
src, out, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
j = json.load(open(src))
tot_old = tot_new = 0
for n, c in j["norm"].items():
    cfg = parse_norm(c)
    old_deg, old_nr = len(c["cheb_coeffs"]) - 1, c["cheb_nr_iters"]
    (d, _), deg, nr, coef, err = fit_site(c["cheb_lo"], c["cheb_hi"], cfg.inv_out_scale, target, 31)
    dep = lambda dg, r: 2 + math.ceil(math.log2(dg)) + 2 * r  # noqa: E731
    tot_old += dep(old_deg, old_nr); tot_new += d
    print(f"{n:24s} ratio {c['cheb_hi'] / c['cheb_lo']:7.1f}  old deg {old_deg:2d}+NR{old_nr} (depth {dep(old_deg, old_nr)})"
          f"  new deg {deg:2d}+NR{nr} (depth {d}) err {err:.1e}")
    c.update(cheb_coeffs=[float(v) for v in coef], cheb_nr_iters=nr)
print(f"total LN depth per token {tot_old} -> {tot_new}")
json.dump(j, open(out, "w"), indent=1)

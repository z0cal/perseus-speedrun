"""Per-plan refresh mix with levels: route, stc_first, input level, output level, cf, raise drop."""
import collections, json, sys
from perseus import _pc
B = [f"block_{b}" for b in range(14)]
PL = {
 "record": ("../../subagent/r4/graphs_combo/r4", "bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm"),
 "em9": ("../lndefer/graphs_combo/lnd_e3", "bootstrap_placements/lnd_fin_e3_em9_cm"),
 "d27sb": ("../lndefer/graphs_combo/lnd_e3", "bootstrap_placements/lnd_fin_e3_d27sb_cm"),
}
out = {}
for name in sys.argv[1:] or PL:
    g, p = PL[name]
    ps = _pc.Program.compile(g, p, blocks=B, keep_going=True)
    rows = []
    for b in B:
        lv = {x[0]: x for x in ps[b].bts_site_levels}
        for x in ps[b].bts_sites:
            r = "stcf" if (x[2] == 0 and x[6]) else ("dense" if x[2] == 0 else str(x[2]))
            l = lv[x[0]]
            rows.append(dict(block=b, site=x[0], kind=x[1], route=r, cf=x[3], real=x[4], drop=x[5], lin=l[1], lout=l[3]))
        print(name, b, ps[b].bts_batches[:3], ps[b].n_ops, ps[b].n_launches, file=sys.stderr)
    out[name] = rows
    mix = collections.Counter(r["route"] for r in rows)
    print(name, len(rows), dict(mix))
    print("  (route,lin->lout):", dict(sorted(collections.Counter((r["route"], r["lin"], r["lout"]) for r in rows).items())))
    print("  ops", sum(ps[b].n_ops for b in B), "launches", sum(ps[b].n_launches for b in B))
json.dump(out, open(sys.argv[0].replace("mix.py", "mix.json"), "w"), indent=0)

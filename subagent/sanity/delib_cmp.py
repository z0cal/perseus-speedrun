"""Sanity: every deliberate refresh of plan vs record, full route tuple (slots, cf, real, raise_drop, stc_first) and the
compiled landing (bts_site_levels). usage (work/perseus): delib_cmp.py GRAPH PLAN GRAPH_REC PLAN_REC"""
import sys
from perseus import _pc
B = [f"block_{b}" for b in range(14)]
def get(g, p):
    ps = _pc.Program.compile(g, p, blocks=B, keep_going=True)
    out = {}
    for b in B:
        lv = {x[0]: x[1:] for x in ps[b].bts_site_levels}
        out[b] = [(x[0], x[2:], lv[x[0]]) for x in ps[b].bts_sites if x[1] == "deliberate"]
    return out
a, r = get(*sys.argv[1:3]), get(*sys.argv[3:5])
for b in B:
    ra = {x[0]: x for x in r[b]}
    for x in a[b]:
        y = ra.get(x[0])
        print(b, x[0], "plan route", x[1], "in/landing", x[2], "| record", (y[1], y[2]) if y else "NO SAME-NAME SITE")

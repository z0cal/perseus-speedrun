"""Sanity: compile 14 chained stages; total check errors, refreshes per stage, every deliberate site (var, route, real),
and the stage-13 route mix. usage (work/perseus, PYTHONPATH=python): sites.py GRAPH_DIR PLAN_DIR"""
import collections, sys
from perseus import _pc
B = [f"block_{b}" for b in range(14)]
g, p = sys.argv[1:3]
ps = _pc.Program.compile(g, p, blocks=B, keep_going=True)
err = 0; tot = 0
for b in B:
    n, e = ps[b].check
    err += len(e)
    s = ps[b].bts_sites
    if b != "block_13":
        tot += len(s)
    if e:
        print(f"  {b}: {len(e)} errors, first: {e[0][:200]}")
print("errors", err, "| refreshes blocks 0-12", tot, "| stage 13", len(ps['block_13'].bts_sites),
      "| per-block", [len(ps[b].bts_sites) for b in B])
d = collections.Counter((b == "block_13", x[2], x[4]) for b in B for x in ps[b].bts_sites if x[1] == "deliberate")
print("deliberate (stage13?, route, real):", dict(d))
print("stage 13 mix:", dict(collections.Counter((x[1], x[2]) for x in ps['block_13'].bts_sites)))
print("sample site tuple:", ps['block_0'].bts_sites[0])

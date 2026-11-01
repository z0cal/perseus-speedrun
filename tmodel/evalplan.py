"""Compile a Decode-8 plan on its combo graph dir (14 stages), report check errors / refresh mix / landings against a
reference plan, save the programs for predict.py. CPU only.
usage (from work/perseus-r4, PYTHONPATH=python): python evalplan.py COMBO PLAN_DIR NAME [REF_COMBO REF_PLAN]"""
import collections, json, os, sys
from perseus import _pc
B = [f"block_{b}" for b in range(14)]
PROGS = "/tmp/claude-1000/-home-zocal-pesquisa-perseus-speedrun/a0e93e83-1a29-462b-88bb-8a12ee7ad2e5/scratchpad/progs"
combo, plan, name = sys.argv[1:4]
ps = _pc.Program.compile(combo, plan, blocks=B, keep_going=True)
os.makedirs(f"{PROGS}/{name}", exist_ok=True)
for b in B:
    ps[b].save(f"{PROGS}/{name}/{b}.json")
errs = sum(len(ps[b].check[1]) for b in B)
mix = collections.Counter(x[2] for b in B for x in ps[b].bts_sites)
print(f"{name}: check errors {errs}, refreshes {sum(mix.values())} {dict(mix)}, per stage {[len(ps[b].bts_sites) for b in B]}")
if len(sys.argv) > 5:
    r = _pc.Program.compile(sys.argv[4], sys.argv[5], blocks=B, keep_going=True)
    dl = sum(1 for b in B for x, y in zip(sorted(r[b].bts_site_levels), sorted(ps[b].bts_site_levels)) if x != y)
    print(f"  landing/input differences vs reference: {dl} sites")

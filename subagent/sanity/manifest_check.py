"""Sanity: compile the 14 stages with the SESSION Config (config=: the stamp-vs-session load-bearing check of
prepare()), then Session.key_manifest (no device): key_needs builds the host-side GPU route plans (StC-first, dense,
SparseB cfgs, classic) with the plan's EvalMod series. usage (work/perseus): manifest_check.py GRAPH_DIR PLAN"""
import json, sys, time
from perseus import _pc
from perseus.serve import gpt2_decode as served
g, plan = sys.argv[1:3]
cfg = served._config(plan, "n32")
print("session bts.evalmod:", repr(cfg["bts.evalmod"]))
B = [f"block_{b}" for b in range(14)]
progs = _pc.Program.compile(g, plan, config=cfg, blocks=B, keep_going=True)
print("compile with config=: errors", sum(len(progs[b].check[1]) for b in B))
keys = _pc.KeySource.generate([], conj=True, layout="packed")
t = time.time()
for regs in (64, 256):
    try:
        spec = json.loads(_pc.Session.key_manifest(cfg, keys, programs=[progs[b] for b in B], regs=regs, pool=16,
                                                   pt_rows=28, stc_cache=8, dyn_rot=[-i for i in range(-1, 32)]))
        json.dump(spec, open(f"/home/zocal/pesquisa/perseus-speedrun/subagent/sanity/spec_{plan.rstrip('/').split('/')[-1]}.json", "w"), indent=1)
        print(f"key_manifest ok (regs={regs}, {time.time() - t:.0f}s): {len(spec['keys'])} keys, contract {spec['contract']}, spru {spec.get('spru')}")
        break
    except Exception as e:  # noqa: BLE001
        print(f"key_manifest regs={regs} raised: {str(e)[:500]}")

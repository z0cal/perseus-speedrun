"""Sanity: the CPU half of speedrun/server.py decode(): Gpt2Handler + manifest() (record the CPU twin on zero rows,
compile the 14 stages against the plan with the session Config, key_needs/key_manifest = host-side GPU route plans).
usage (work/perseus, local_env): prepare_cpu.py PLAN WORK_DIR"""
import collections, glob, json, os, sys, time, types
from perseus.client import contract_hash
from perseus.serve import gpt2_decode as served
plan, work = sys.argv[1], sys.argv[2]
args = types.SimpleNamespace(plan=plan, chain="n32", layers=None, packing="cachemir_complex",
                             weights=os.environ["WEIGHTS_PATH"], key_layout="packed", pt_rows=28, stc_cache=8,
                             work_dir=work)
t0 = time.time(); print("start", flush=True)
h = served.Gpt2Handler(args); print("handler built", flush=True); h.run_.probe = int(os.environ.get("PROBE", "4"))
print("cfg bts.evalmod:", repr(h.run_.cfg["bts.evalmod"]))
man = h.manifest({"tokens": int(os.environ.get("PROBE", "4")), "argmax": True, "contract": contract_hash(served._config(plan, "n32"))})
spec = json.loads(man["spec"])
print("manifest ok in %.0fs: input %s, %d switching keys, contract %s" % (time.time() - t0, man["input"], len(spec["keys"]), spec["contract"]))
r = h.run_
tot = 0
for i, lab in enumerate(r.labels):
    p = r.progs.get(lab)
    if p is None:
        print(lab, "no program"); continue
    n, e = p.check
    s = p.bts_sites
    mix = collections.Counter((x[1], x[2]) for x in s)
    if i < 13:
        tot += len(s)
    print(f"{lab}: bts={len(s)} errors={len(e)} mix={dict(mix)}")
print("blocks 0-12 refreshes:", tot)
c = collections.Counter()
for f in sorted(glob.glob(os.path.join(work, "ref", "block_*", "graph.json"))):
    for nd in json.load(open(f))["nodes"]:
        if nd.get("op_type") == "deliberate_bootstrap":
            c[(os.path.basename(os.path.dirname(f)) == "block_13", nd.get("output_level"), nd.get("output_noise_level"))] += 1
print("twin-recorded deliberate landings (stage13?, level, deg):", dict(c))

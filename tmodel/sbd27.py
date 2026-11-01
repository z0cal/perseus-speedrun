"""Copy a Decode-8 plan dir with its SparseB refreshes on the d27 K=24 table and one more raise drop each (runtime
bts.evalmod=d27sb, base 512:+1, variants +6): the landings stay the plan's. usage: python -I sbd27.py SRC_DIR DST_DIR"""
import glob, json, os, re, shutil, sys
src, dst = sys.argv[1:3]
shutil.copytree(src, dst)
for f in sorted(glob.glob(dst + "/block_*_placement.json")):
    p = json.load(open(f))
    rd = p.setdefault("raise_drop", {})
    for v, s in p.get("sparse_slots", {}).items():
        if s == 512 and v in rd:
            rd[v] += 1
    json.dump(p, open(f, "w"), indent=2)
t = open(dst + "/PLAN_CMD.txt").read()
m = re.search(r"FIDESLIB_BTS_RAISE_DROP=512:(\d+)", t)
t = t.replace(m.group(0), f"FIDESLIB_BTS_RAISE_DROP=512:{int(m.group(1)) + 1}")
t = t.replace("FIDESLIB_BTS_RAISE_DROPS=1,2,3,4,5 ", "FIDESLIB_BTS_RAISE_DROPS=1,2,3,4,5,6 ")
t = re.sub(r"^(runtime=.*)$", r"\1 FIDESLIB_BTS_EVALMOD=d27sb", t, count=1, flags=re.M)
t = f"variant={os.path.basename(src)} with the SparseB refreshes on the d27 K24 table (d27sb) and one more raise drop each: same landings (tmodel/sbd27.py)\n" + t
open(dst + "/PLAN_CMD.txt", "w").write(t)

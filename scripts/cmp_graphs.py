"""Compare two capture directories block by block: op structure (op, output, inputs, level) and magnitudes."""
import glob, json, os, sys
a, b = sys.argv[1:3]
for f in sorted(glob.glob(f"{b}/block_*/graph.json"), key=lambda p: int(p.split("block_")[-1].split("/")[0])):
    blk = f.split("/")[-2]
    g = f"{a}/{blk}/graph.json"
    if not os.path.exists(g):
        print(blk, "missing")
        continue
    na, nb = json.load(open(g))["nodes"], json.load(open(f))["nodes"]
    key = lambda n: (n["op_type"], n["output"], tuple(n["inputs"]), n["output_level"])
    same = [key(x) for x in na] == [key(x) for x in nb]
    dm = max((abs(x.get("output_max_abs", 0) - y.get("output_max_abs", 0)) / max(abs(y.get("output_max_abs", 0)), 1e-30)
              for x, y in zip(na, nb)), default=0.0)
    print(f"{blk:9s} nodes {len(na):5d}/{len(nb):5d} structure {'SAME' if same else 'DIFF'} max rel mag diff {dm:.2e}")

"""Per-plan route mix (blocks 0-12) and a latency proxy with the planner's default bts_ms (dense 28.77, SparseB-512
18.59, 1-slot 16.10 ms; old GPU, relative only)."""
import json, sys, os
MS = {0: 28.77, 512: 18.59, 1: 16.10}
W = os.path.expanduser("~/pesquisa/perseus-speedrun/work/perseus/bootstrap_placements")
for cand in sys.argv[1:]:
    n = {0: 0, 512: 0, 1: 0}; drops = 0
    for b in range(13):
        d = json.load(open(f"{W}/{cand}/block_{b}_placement.json"))
        sp = d["sparse_slots"]
        tot = d["summary"]["total_bootstraps"]
        k512 = sum(1 for v in sp.values() if v == 512); k1 = sum(1 for v in sp.values() if v == 1)
        n[512] += k512; n[1] += k1; n[0] += tot - k512 - k1
        drops += sum(d["raise_drop"].values())
    ms = sum(MS[k] * v for k, v in n.items())
    print(f"{cand:36s} total {sum(n.values()):3d}  dense {n[0]:3d}  sparse512 {n[512]:3d}  spru {n[1]:3d}  ~{ms:6.0f} ms  sum(raise_drop) {drops}")

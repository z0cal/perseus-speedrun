"""Audit: compile a plan dir against a capture graph dir with the build's _pc (CPU only) and list the compile-check
errors per block (the [plan_level_error] etc. the served decode would raise at prepare()).
usage (from inside the repo, PYTHONPATH=python): python compile_check.py GRAPH_DIR PLAN_DIR [block labels...]"""
import sys
import time

from perseus import _pc

g, p = sys.argv[1], sys.argv[2]
blocks = sys.argv[3:] or [f"block_{b}" for b in range(13)]
t = time.time()
try:
    progs = _pc.Program.compile(g, p, blocks=blocks, keep_going=True)
except Exception as e:  # noqa: BLE001
    print("COMPILE RAISED:", str(e)[:3000])
    sys.exit(1)
tot_err = 0
for lab in blocks:
    pr = progs.get(lab)
    if pr is None:
        print(lab, "missing")
        continue
    nchk, errs = pr.check
    n = len(errs)
    tot_err += n
    print(f"{lab}: bts={pr.n_bootstraps} ops={pr.n_ops} checks={nchk} errors={n}", *(["  " + e for e in errs[:4]] if n else []), sep="\n" if n else " ")
print(f"total check errors {tot_err}  ({time.time() - t:.0f} s)")

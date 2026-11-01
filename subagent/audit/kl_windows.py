"""Audit: float64 KL (scripts/kl_eval.py's model) of the final config and the record's on given pool windows.
usage (in work/perseus-rec, PYTHONPATH=python): python kl_windows.py OFFSET [OFFSET...]   (16 tokens from each offset)
       python kl_windows.py --docstarts N   (the first N document starts of the pool)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/pesquisa/perseus-speedrun/scripts"))
import kl_eval  # noqa: E402
from perseus.impl import config as pconfig  # noqa: E402

W = os.path.expanduser("~/pesquisa/perseus-speedrun")
CFG = {"final": pconfig.load_configs(os.environ.get("KLW_CFG", f"{W}/configs/cand_dfr13_cm/configs.json")),
       "record": pconfig.load_configs(f"{W}/work/perseus-rec/python/perseus/configs/model/approximation/"
                                      "gpt2_base_n32_kc16_cm/configs.json")}
pool = np.load(kl_eval.POOL, mmap_mode="r")
if sys.argv[1] == "--docstarts":
    e = np.where(np.asarray(pool) == 50256)[0]
    offs = [int(o) + 1 for o in e[:int(sys.argv[2])]]
else:
    offs = [int(x) for x in sys.argv[1:]]
res = {k: [] for k in CFG}
for o in offs:
    w = np.array(pool[o:o + 16]).astype(np.int64)
    lp = kl_eval.logsoftmax(kl_eval.forward_exact(w))
    row = []
    for name, cf in CFG.items():
        with np.errstate(all="ignore"):
            lq = kl_eval.logsoftmax(kl_eval.forward_approx(w, cf))
        k = (np.exp(lp) * (lp - lq)).sum(-1)[8:16]
        km = float(np.mean(k)) if np.all(np.isfinite(k)) else float("inf")
        res[name].append(km)
        row.append(f"{name} {km:.3e}")
    if len(offs) <= 20 or not all(np.isfinite(res["final"][-1:])) or res["final"][-1] > 1e-4:
        print(o, " | ".join(row), flush=True)
for name, v in res.items():
    v = np.array(v)
    fin = np.isfinite(v)
    print(f"{name}: windows {len(v)}, non-finite {int((~fin).sum())}, mean(finite) {v[fin].mean() if fin.any() else float('nan'):.3e}, "
          f"max(finite) {v[fin].max() if fin.any() else float('nan'):.3e}, >1e-3: {int((v > 1e-3).sum())}")

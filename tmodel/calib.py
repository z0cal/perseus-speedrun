"""Calibrate tau/kappa/eps (ms per unit) of units.py on the pc-bench stage times (RTX PRO 6000 WK, n32, K = 1,
gpu/results-2026-10-30-runpod). usage: python -I calib.py  -> writes calib.json next to it."""
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import units as U  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
EM = U.load_em(f"{H}/em/n32_t45.jsonl")
ch = "n32"
rows = [  # (name, units, measured ms)
    ("raise", U.raise_(54), 0.55),
    ("cts0", U.lt_stage(54, 16, 2, 31, ch, giant_identity=True), 2.31),
    ("cts1", U.lt_stage(52, 16, 2, 31, ch), 2.34),
    ("cts2", U.lt_stage(50, 16, 2, 31, ch), 2.23),
    ("cts3", U.lt_stage(48, 8, 2, 15, ch), 1.38),
    ("em_shipped_x2", U.add(U.z(), U.evalmod(EM["Shipped_K16"]["oplist"], ch), 2), 8.20),
    ("em_d27_x2", U.add(U.z(), U.evalmod(EM["D27_K16"]["oplist"], ch), 2), 10.40),
    ("stc0", U.lt_stage(26, 16, 4, 63, ch), 1.61),
    ("stc1", U.lt_stage(24, 16, 4, 63, ch), 1.52),
    ("stc2", U.lt_stage(22, 16, 4, 63, ch, offset=True), 1.83),
]
A = np.array([r[1] for r in rows])
y = np.array([r[2] for r in rows])
w = 1.0 / y  # relative residuals
x, _ = nnls(A * w[:, None], y * w)
pred = A @ x
print("tau (ms/NTT-limb) %.3g  kappa (ms/conv-unit) %.3g  eps (ms/EW-limb) %.3g" % tuple(x))
for (n, u, m), p in zip(rows, pred):
    print(f"  {n:14s} meas {m:6.2f}  model {p:6.2f}  ({100 * (p / m - 1):+5.1f} %)   units ntt {u[0]:7.0f} conv {u[1]:7.0f} ew {u[2]:7.0f}")
json.dump({"tau": x[0], "kappa": x[1], "eps": x[2]}, open(f"{H}/calib.json", "w"))

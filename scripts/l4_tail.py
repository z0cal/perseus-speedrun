"""Tail of the centred scores of layer 4 off the calibration set: the go / no-go of candidate 1 (layer 4 with delta2 = 4,
`scripts/sm_l4_delta2.py`). With margin m the layer's softmax stays a softmax only while every score u = s - s_ref
stays below ~tol(m) x U* (sm_l4_delta2.py's tolerance column: 1.15 at m = 1.00, 1.20 at 1.05, 1.25 at 1.10, 1.56 for
the record); above it the Goldschmidt leaves its band and the row is garbage or NaN.

Runs float64 GPT-2 (fit_gpt2_decode16.taps, the calibration's own code) on 16-token windows that the calibration did NOT
see: OpenWebText shard 0042 documents outside fit_gpt2_decode16_ref's 6000 (seed 1234), and optionally C4 validation.
Uses the record config's per-head references for layer 4 and reports, per head and overall, the max u / U* and how
many windows exceed each threshold.

usage (on the machine with the data):
  python scripts/l4_tail.py --perseus <clone> --calib [--owt 0042.parquet] [--gpt2 SNAPSHOT_DIR] [--docs 20000] [--c4 FILE]
  python scripts/l4_tail.py --perseus <clone> --selftest        (random weights; checks the plumbing only)"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

THRESH = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.40, 1.56)
LAYER = 4


def ref_index(ref, kc):
    return {"sink": 0, "self": kc - 1, "prev": max(kc - 2, 0)}[ref]


def scan(F, w, ids_list, refs, batch=50):
    """max u per head [12] and the per-window max over heads (one value per window) for layer LAYER."""
    head_max = np.full(12, -np.inf)
    win_max = []
    for ids in ids_list:
        for s0 in range(0, len(ids), batch):
            scores = [dict() for _ in range(12)]
            F.taps(w, ids[s0:s0 + batch], scores, [0.0] * 12)
            per_win = np.full(len(ids[s0:s0 + batch]), -np.inf)
            for kc, rows in scores[LAYER].items():
                s = rows[0]                                    # [B, H, kc]
                idx = np.array([ref_index(r, kc) for r in refs])
                sref = np.take_along_axis(s, idx[None, :, None].repeat(s.shape[0], 0), axis=-1)
                u = (s - sref).max(-1)                         # [B, H]
                head_max = np.maximum(head_max, u.max(0))
                per_win = np.maximum(per_win, u.max(1))
            win_max.append(per_win)
    return head_max, np.concatenate(win_max)


def report(name, head_max, win_max, ustar):
    print(f"\n== {name}: {len(win_max)} janelas")
    print("   max u / U* por cabeça: " + " ".join(f"{x / ustar:.3f}" for x in head_max))
    print(f"   max geral: {win_max.max():.3f} = {win_max.max() / ustar:.3f} U*")
    for t in THRESH:
        n = int(np.sum(win_max > t * ustar))
        print(f"   janelas com u > {t:.2f} U*: {n} ({100 * n / len(win_max):.4f} %)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perseus", required=True, help="the perseus clone (upstream d97a6f7 or a branch on it)")
    ap.add_argument("--owt")
    ap.add_argument("--gpt2")
    ap.add_argument("--docs", type=int, default=20000, help="held-out OWT documents to scan")
    ap.add_argument("--c4", help="C4 validation .json.gz (optional, another distribution)")
    ap.add_argument("--calib", action="store_true", help="also scan the calibration windows (must give max = 1.000 U*)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.perseus, "scripts", "utils"))
    sys.path.insert(0, os.path.join(a.perseus, "python"))
    import fit_gpt2_decode16 as F  # noqa: E402
    cfg = json.load(open(os.path.join(a.perseus, "python/perseus/configs/model/approximation/"
                                      "gpt2_base_n32_kc16r_cm/configs.json")))
    s4 = cfg["softmax"][f"transformer.h.{LAYER}.attn"]
    refs, ustar = s4["ref_keys"], s4["clip_hi"] / 1.25
    print(f"camada {LAYER}: U* = {ustar:.4f} (clip_hi / 1,25), referências {refs}")

    if a.selftest:
        rng = np.random.default_rng(0)
        V, D = 1000, 768
        w = {"wte.weight": rng.normal(0, 0.02, (V, D)), "wpe.weight": rng.normal(0, 0.01, (1024, D))}
        for i in range(12):
            p = f"h.{i}."
            w.update({p + "ln_1.weight": np.ones(D), p + "ln_1.bias": np.zeros(D),
                      p + "attn.c_attn.weight": rng.normal(0, 0.02, (D, 3 * D)), p + "attn.c_attn.bias": np.zeros(3 * D),
                      p + "attn.c_proj.weight": rng.normal(0, 0.02, (D, D)), p + "attn.c_proj.bias": np.zeros(D),
                      p + "ln_2.weight": np.ones(D), p + "ln_2.bias": np.zeros(D),
                      p + "mlp.c_fc.weight": rng.normal(0, 0.02, (D, 4 * D)), p + "mlp.c_fc.bias": np.zeros(4 * D),
                      p + "mlp.c_proj.weight": rng.normal(0, 0.02, (4 * D, D)), p + "mlp.c_proj.bias": np.zeros(D)})
        ids = rng.integers(0, V, (120, 16))
        hm, wm = scan(F, w, [ids], refs)
        report("selftest (pesos aleatórios: só confere o encanamento)", hm, wm, ustar)
        return

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    hub = os.path.join(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "hub")
    snap = a.gpt2 or os.path.dirname(sorted(glob.glob(os.path.join(
        hub, "models--openai-community--gpt2", "snapshots", "*", "model.safetensors")))[0])
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    w = F.load_safetensors(os.path.join(snap, "model.safetensors"))
    owt = pq.read_table(F.data_file(F.OWT, a.owt), columns=["text"]).column("text").to_pylist()
    if a.calib:
        c_start, c_rand = F.windows(owt, 1234, 6000, 4000, 160, tok)                 # fit_gpt2_decode16_ref's windows
        hm, wm = scan(F, w, [c_start, c_rand], refs)
        report("calibração (deve dar max = 1,000 U*)", hm, wm, ustar)
    seen = set(np.random.default_rng(1234).permutation(len(owt))[:6000].tolist())   # fit_gpt2_decode16_ref's docs
    rest = [t for i, t in enumerate(owt) if i not in seen]
    o_start, o_rand = F.windows(rest, 4321, a.docs, 4000, 160, tok)
    hm, wm = scan(F, w, [o_start, o_rand], refs)
    report(f"OWT 0042 fora da calibração ({a.docs} documentos: inícios + offsets aleatórios)", hm, wm, ustar)
    if a.c4:
        import gzip
        texts = [json.loads(l)["text"] for l in gzip.open(a.c4, "rt")]
        c_start, c_rand = F.windows(texts, 4321, min(a.docs, len(texts)), 4000, 160, tok)
        hm, wm = scan(F, w, [c_start, c_rand], refs)
        report(f"C4 {os.path.basename(a.c4)}", hm, wm, ustar)


if __name__ == "__main__":
    main()

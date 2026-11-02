"""search.py with extra knobs (subagent, 2026-10-27). Same pipeline (kl_eval + capture_rec.sh + plan_count.sh);
results go to subagent/ganhos-rapidos/logs/search2.tsv (search.tsv is left alone).

New knobs (on top of every search.py knob, applied AFTER them, so `smsplit=` composes):
  smwide=R:DEG[:PAD]  softmax exp: per layer r = min(R, l1) squarings fewer; the exp series refit at degree DEG on an
                      interval 2^r wider (the SM_FOLD input map is unchanged: cheb_a/sf stays the same);
                      PAD widens the fit interval by that factor (robustness to scores past the clip), default 1.
  smrel=AMAX2[:DEG[:PAD]]  softmax exp, RELATIVE (IRLS) fit: per layer r = 2 squarings fewer when the exp half-width
                      a = cheb_b <= AMAX2, else r = 1 (r <= l1); degree DEG (default 14, BSGS depth 4 = the record's
                      degree 8); PAD widens the fit interval (scores past the clip).
  gelubs              GELU refit (the record's own IRLS fit) at the largest degree whose BSGS depth is one lower:
                      trimmed degree 62 -> 60 (depth 7 -> 6), 254 -> 248 (9 -> 8); others untouched.
  gelutr              same degrees, but the shipped series TRUNCATED (no refit).
  gelu94=D            layers 1, 3, 10 (degree 94, depth 7) refit at D (60 = depth 6).

    python -I search2.py NAME KNOB [KNOB ...] [--no-plan]
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.expanduser("~/pesquisa/perseus-speedrun/scripts")
sys.path.insert(0, SCR)
import search  # noqa: E402  (scripts/search.py: build, constants)

W, REC = search.W, search.REC
TSV = f"{HERE}/logs/search2.tsv"
GELU_TARGET = {62: 60, 254: 248}


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def fit_gelu_irls(X, degree):
    """scripts/utils/fit_gpt2_decode16.fit_gelu's fit (X already the shipped 1.3 x max)."""
    m = max(8 * (degree + 1), 4000)
    t = np.cos(np.pi * (np.arange(m) + 0.5) / m)
    f = gelu(X * t) / X
    c = np.polynomial.chebyshev.chebfit(t, f, degree)
    for _ in range(6):
        r = np.abs(np.polynomial.chebyshev.chebval(t, c) - f)
        c = np.polynomial.chebyshev.chebfit(t, f, degree, w=np.sqrt(r / r.max() + 0.02))
    tt = np.linspace(-1, 1, 20001)
    e = float(np.abs(np.polynomial.chebyshev.chebval(tt, c) - gelu(X * tt) / X).max())
    return c, e


def trimmed_degree(c, eps):
    from perseus.impl import poly
    return len(poly._cheb_trim(c, eps)) - 1


def apply_gelu(c, mode, only=None, deg_to=None):
    for k, g in c["softgelu"].items():
        l = int(k.split(".")[2])
        if only is not None and l not in only:
            continue
        d = trimmed_degree(g["cheb_coeffs"], g.get("cheb_eps"))
        tgt = deg_to if deg_to is not None else GELU_TARGET.get(d)
        if tgt is None:
            continue
        if mode == "tr":
            g["cheb_coeffs"] = list(g["cheb_coeffs"][:tgt + 1])
            g["note"] = f"truncated {d} -> {tgt} (search2 gelutr)"
        else:
            cf, e = fit_gelu_irls(g["xmax"], tgt)
            g["cheb_coeffs"] = list(cf)
            g["note"] = f"refit degree {tgt} (was {d}), max err {e:.2e} (search2 gelubs)"


def apply_smwide(c, R, deg, pad=1.0):
    for l in range(12):
        s = c["softmax"][f"transformer.h.{l}.attn"]
        l1 = int(s["n_squarings"])
        r = min(R, l1)
        if r == 0:
            continue
        a = float(s["cheb_b"]) * 2 ** r                    # new half-width in t units of exp(a t)
        A = a * pad
        n = max(8 * (deg + 1), 400)
        t = np.cos(np.pi * (np.arange(n) + 0.5) / n)
        cf = np.polynomial.chebyshev.chebfit(t, np.exp(A * t), deg)
        s["n_squarings"] = l1 - r
        s["delta1"] = 2 ** (l1 - r)
        s["cheb_a"], s["cheb_b"] = -A, A
        s["cheb_coeffs"] = list(cf)
        s["note"] = s.get("note", "") + f" | smwide r={r} deg={deg} pad={pad} (search2)"


def fit_exp_rel(A, deg, pad=1.0, iters=12):
    """exp(A t) on [-pad, pad] (returned in the [-1, 1] variable of the padded interval), IRLS toward relative minimax."""
    from numpy.polynomial import chebyshev as C
    n = max(16 * (deg + 1), 2000)
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    f = np.exp(A * pad * t)
    w = 1.0 / f
    c = C.chebfit(t, f, deg, w=w)
    for _ in range(iters):
        r = np.abs((C.chebval(t, c) - f) / f)
        c = C.chebfit(t, f, deg, w=w * np.sqrt(r / r.max() + 0.02))
    return c


def apply_smrel(c, amax2, deg=14, pad=1.0):
    for l in range(12):
        s = c["softmax"][f"transformer.h.{l}.attn"]
        l1 = int(s["n_squarings"])
        a0 = float(s["cheb_b"])
        r = min(2 if a0 <= amax2 else 1, l1)
        if r == 0:
            continue
        A = a0 * 2 ** r
        cf = fit_exp_rel(A, deg, pad)
        s["n_squarings"] = l1 - r
        s["delta1"] = 2 ** (l1 - r)
        s["cheb_a"], s["cheb_b"] = -A * pad, A * pad
        s["cheb_coeffs"] = list(cf)
        s["note"] = s.get("note", "") + f" | smrel r={r} deg={deg} pad={pad} (search2)"


def build2(knobs):
    base = [k for k in knobs if k.split("=")[0] in ("gelu", "ln", "smtgt", "smsplit", "smtop")
            or k.startswith("gelu.") or k.startswith("ln.")]
    c = search.build(base)
    for kn in knobs:
        if kn in base:
            continue
        k, _, v = kn.partition("=")
        if k == "smwide":
            p = v.split(":")
            apply_smwide(c, int(p[0]), int(p[1]), float(p[2]) if len(p) > 2 else 1.0)
        elif k == "smrel":
            p = v.split(":")
            apply_smrel(c, float(p[0]), int(p[1]) if len(p) > 1 else 14, float(p[2]) if len(p) > 2 else 1.0)
        elif k == "gelubs":
            apply_gelu(c, "fit")
        elif k == "gelutr":
            apply_gelu(c, "tr")
        elif k == "gelu94":
            apply_gelu(c, "fit", only=(1, 3, 10), deg_to=int(v))
        else:
            raise SystemExit(f"unknown knob {kn}")
    return c


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name, knobs = args[0], args[1:]
    cdir = f"{REC}/python/perseus/configs/model/approximation/cand_{name}"
    os.makedirs(cdir, exist_ok=True)
    cfg_path = f"{cdir}/configs.json"
    json.dump(build2(knobs), open(cfg_path, "w"), indent=1)
    t0 = time.time()
    if "--plan-only" not in sys.argv:
        import kl_eval
        from perseus.impl import config as pconfig
    kl_mean, kl_max = (0.0, 0.0) if "--plan-only" in sys.argv else \
        kl_eval.Evaluator("owt", 8).kl(pconfig.load_configs(cfg_path))
    print(f"KL {kl_mean:.3e} max {kl_max:.3e}", flush=True)
    bts = "-"
    if "--no-plan" not in sys.argv and math.isfinite(kl_mean):
        cap = f"{W}/caps/{name}"
        r = subprocess.run(["bash", f"{W}/capture_rec.sh", cap,
                            f"configs/model/approximation/cand_{name}/configs.json"],
                           capture_output=True, text=True)
        if r.returncode:
            bts = "capture-failed"
            open(f"{HERE}/{name}.capture.log", "w").write(r.stdout + r.stderr)
        else:
            out = subprocess.run(["bash", f"{W}/plan_count.sh", cap, f"cand_{name}"], capture_output=True, text=True)
            bts = out.stdout.strip().split()[-1] if out.returncode == 0 else "plan-failed"
            if out.returncode:
                open(f"{HERE}/{name}.plan.log", "w").write(out.stdout + out.stderr)
    line = f"{name}\t{' '.join(knobs)}\t{kl_mean:.3e}\t{kl_max:.3e}\t{bts}\tmincut\t{time.time() - t0:.0f}s"
    print(line)
    with open(TSV, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()

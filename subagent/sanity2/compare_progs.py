"""sanity2: o programa compilado do grafo gravado pelo gemeo do servidor (d27) == o do grafo capturado em que o plano
foi feito? Compila os dois com a Config da sessao (served._config) e compara bloco a bloco.
uso (work/perseus-r4): compare_progs.py PLAN DIR_A DIR_B [OUT_DIR]"""
import hashlib, json, os, sys
from perseus import _pc
from perseus.serve import gpt2_decode as served
plan, A, B = sys.argv[1:4]
out = sys.argv[4] if len(sys.argv) > 4 else None
cfg = served._config(plan, "n32")
blocks = [f"block_{b}" for b in range(14)]
pa = _pc.Program.compile(A, plan, config=cfg, blocks=blocks, keep_going=True)
pb = _pc.Program.compile(B, plan, config=cfg, blocks=blocks, keep_going=True)


def sig(p):
    d = {}
    for k in ("n_ops", "n_bootstraps", "n_finishes", "n_launches", "n_pairs", "bts_sites", "bts_site_levels", "inputs",
              "outputs", "pt_refs", "check", "coeff_sums", "bts_batches", "pt_keys", "scalars", "rotations"):
        try:
            v = getattr(p, k)
            v = v() if callable(v) else v
            d[k] = json.loads(json.dumps(v, default=repr))
        except Exception as e:  # noqa: BLE001
            d[k] = f"<{type(e).__name__}: {str(e)[:80]}>"
    try:
        d["rot_steps"] = sorted(p.rot_steps())
    except Exception as e:  # noqa: BLE001
        d["rot_steps"] = f"<{e}>"
    return d


tot = {"same": 0, "diff": 0}
for lab in blocks:
    a, b = pa.get(lab), pb.get(lab)
    sa, sb = sig(a), sig(b)
    diffk = [k for k in sa if sa[k] != sb[k]]
    sv = ""
    if out:
        os.makedirs(out, exist_ok=True)
        fa, fb = os.path.join(out, f"{lab}_A.prog"), os.path.join(out, f"{lab}_B.prog")
        try:
            a.save(fa); b.save(fb)
            ha = hashlib.sha256(open(fa, "rb").read()).hexdigest()[:16]; hb = hashlib.sha256(open(fb, "rb").read()).hexdigest()[:16]
            sv = f" save(): {'IDENTICAL' if ha == hb else 'DIFFER'} ({ha} {hb})"
        except Exception as e:  # noqa: BLE001
            sv = f" save() failed: {str(e)[:120]}"
    tot["same" if not diffk else "diff"] += 1
    print(f"{lab}: errors A={len(sa['check'][1]) if isinstance(sa['check'], list) else sa['check']} B={len(sb['check'][1]) if isinstance(sb['check'], list) else sb['check']}; "
          f"bts A={sa['n_bootstraps']} B={sb['n_bootstraps']}; ops A={sa['n_ops']} B={sb['n_ops']}; differing fields: {diffk}{sv}")
    for k in diffk[:3]:
        print(f"    {k}: A={str(sa[k])[:300]}\n    {' ' * len(k)}  B={str(sb[k])[:300]}")
print(tot)

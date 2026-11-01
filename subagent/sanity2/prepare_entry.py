"""sanity2: a metade de CPU do speedrun/server.py decode() para um plano: Gpt2Handler + manifest() (grava o gemeo de
CPU com linhas zero, compila os 14 estagios contra o plano com a Config da sessao, key_manifest sem device) e compara
os fluxos gravados com o grafo em que o plano foi feito.
uso (de work/perseus-r4, PYTHONPATH=python): prepare_entry.py PLAN WORK_DIR GRAPH_DIR [GRAPH_DIR_STAGE13]"""
import collections
import json
import os
import sys
import time
import types

plan, work, gdir = sys.argv[1], sys.argv[2], sys.argv[3]
g13 = sys.argv[4] if len(sys.argv) > 4 else gdir
t0 = time.time()

# --- a ordem de import do servidor: speedrun.record1, perseus.client, perseus.serve.gpt2_decode
from speedrun.record1 import PROMPT, pack, unpack  # noqa: F401,E402
from perseus.client import contract_hash  # noqa: E402
from perseus.serve import gpt2_decode as served  # noqa: E402
from perseus.examples.gpt2 import env  # noqa: E402

_orig_export = env.export_env
seen = {}


def _spy(*a, **kw):
    seen["block_imported_before_export"] = "perseus.examples.gpt2.block" in sys.modules
    seen["model_imported_before_export"] = "perseus.examples.gpt2.model" in sys.modules
    seen["norm_imported_before_export"] = "perseus.impl.norm" in sys.modules
    seen["LN_DEFER_before"] = os.environ.get("LN_DEFER")
    seen["overrides_LN_DEFER"] = kw.get("LN_DEFER")
    seen["overrides_EVALMOD"] = kw.get("FIDESLIB_BTS_EVALMOD")
    r = _orig_export(*a, **kw)
    seen["LN_DEFER_after"] = os.environ.get("LN_DEFER")
    seen["EVALMOD_after"] = os.environ.get("FIDESLIB_BTS_EVALMOD")
    seen["CHAIN_PRESET_after"] = os.environ.get("CHAIN_PRESET")
    return r


env.export_env = _spy
if os.environ.get("TWIN_NO_LNDEFER") == "1":
    # what if LN_DEFER does not reach the twin (the compile Config keeps model.ln_defer): does prepare() fail loudly?
    _opr2 = env.plan_runtime

    def _pr2(*a, **k):
        d = dict(_opr2(*a, **k))
        d.pop("LN_DEFER", None)
        return d

    env.plan_runtime = _pr2
if os.environ.get("TWIN_NO_EVALMOD") == "1":
    # hypothesis test: the twin (export_env + fakecap.session_env read env.plan_runtime) WITHOUT the EvalMod switch,
    # the compile Config (served._config = pcfg.plan_runtime) still WITH it
    _opr = env.plan_runtime

    def _pr(*a, **k):
        d = dict(_opr(*a, **k))
        d.pop("FIDESLIB_BTS_EVALMOD", None)
        return d

    env.plan_runtime = _pr
LOWMEM = os.environ.get("LOWMEM", "0") == "1"
bw_flags = {}
if LOWMEM:
    # (1) only the current block's weights live (the others are rebuilt when asked: same values, same names);
    # (2) the recorder drops the plaintext vectors (prepare/compile/key_manifest read nodes, objs, metas only)
    from perseus.examples.gpt2.model import Gpt2Model
    from perseus.impl import driver as _drv
    _orig_bw = Gpt2Model.block_weights

    def _bw_lowmem(self, b):
        for k in list(self._bw):
            if k != b:
                del self._bw[k]
        w = _orig_bw(self, b)
        bw_flags[b] = getattr(w, "ln_defer", None)
        return w

    Gpt2Model.block_weights = _bw_lowmem
    _orig_br = _drv.BlockRec.__init__

    def _br(self, nodes, pts, objs, metas):
        _orig_br(self, nodes, {k: None for k in pts}, objs, metas)

    _drv.BlockRec.__init__ = _br
    # (3) structure-only twin (CapCt.novec): plaintext and ciphertext values kept as ONE slot, copied (upstream keeps
    # a view of the full buffer, and the slot-pt store keeps every full weight vector). Tags are computed from the
    # full vector before the Pt is built (pt_tag(v)), so the recorded structure does not change.
    import numpy as _np
    from perseus.impl import fakecap as _FC
    _opt = _FC.Pt.__init__

    def _pt_init(self, vec, level, deg=1, tag=None, src=("values", None)):
        if _FC.CapCt.novec and isinstance(vec, _np.ndarray) and vec.ndim == 1 and vec.shape[0] > 1:
            vec = vec[:1].copy()
        _opt(self, vec, level, deg, tag, src)

    _FC.Pt.__init__ = _pt_init
    _oct = _FC.CapCt.__init__

    def _ct_init(self, *a, **k):
        _oct(self, *a, **k)
        if _FC.CapCt.novec and isinstance(self.vec, _np.ndarray) and self.vec.base is not None:
            self.vec = self.vec.copy()

    _FC.CapCt.__init__ = _ct_init
    _oss = _FC.CapInf.set_slot_pt

    def _ssp(self, name, values, level=0, deg=1):
        _oss(self, name, values, level, deg)
        if _FC.CapCt.novec and name in self.slot_pts and getattr(self.slot_pts[name], "shape", (0,))[0] > 1:
            self.slot_pts[name] = self.slot_pts[name][:1].copy()

    _FC.CapInf.set_slot_pt = _ssp
args = types.SimpleNamespace(plan=plan, chain="n32", layers=None, packing="cachemir_complex",
                             weights=os.environ["WEIGHTS_PATH"], key_layout="packed", pt_rows=28, stc_cache=8,
                             work_dir=work)
print("start", flush=True)
h = served.Gpt2Handler(args)
from perseus.examples.gpt2 import block as _block  # noqa: E402
print("handler built %.0fs" % (time.time() - t0), flush=True)
print("export_env spy:", seen, flush=True)
print("block.LN_DEFER_EXIT =", _block.LN_DEFER_EXIT, flush=True)
r = h.run_
fhe = r.rec.model.inf.fhe
print("twin fakecap landing:", getattr(fhe, "landing", None), flush=True)
print("session cfg: bts.evalmod=%r model.ln_defer=%r chain.preset=%r" % (r.cfg["bts.evalmod"], r.cfg["model.ln_defer"],
                                                                         r.cfg["chain.preset"]), flush=True)
probe = int(os.environ.get("PROBE", "4"))
r.probe = probe
steps = int(os.environ.get("STEPS", "16"))
try:
    man = h.manifest({"tokens": steps, "argmax": True, "contract": contract_hash(served._config(plan, "n32"))})
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print("MANIFEST FAILED after %.0fs: %s" % (time.time() - t0, str(e)[:2000]), flush=True)
    man = None
if man is not None:
    spec = json.loads(man["spec"])
    with open(os.path.join(work, "keyspec.json"), "w") as f:
        json.dump(spec, f, indent=1)
    print("manifest ok in %.0fs: input %s, %d switching keys, contract %s" % (time.time() - t0, man["input"],
                                                                             len(spec["keys"]), spec["contract"]))
    tot = 0
    for i, lab in enumerate(r.labels):
        p = r.progs.get(lab)
        if p is None:
            print(lab, "no program")
            continue
        n, e = p.check
        s = p.bts_sites
        mix = collections.Counter((x[1], x[2]) for x in s)
        if i < 13:
            tot += len(s)
        print(f"{lab}: bts={len(s)} check_errors={len(e)} mix={dict(mix)}")
    print("blocks 0-12 refreshes:", tot)
# block weights LN flag
bw = getattr(r.rec.model, "_bw", {})
print("block weights ln_defer flags:", {b: getattr(w, "ln_defer", None) for b, w in sorted(bw.items())}, bw_flags)


# --- graph comparison: twin's recorded streams vs the capture the plan was made on (inputs canonical: position among
# the block's inputs, as driver.diff_block); also driver.diff_block(capture, twin) = what PlannedRun tolerates
from perseus.impl.driver import BlockRec as _BR, diff_block as _diff, _canon


def ckey(nd, pos):
    a = dict(nd.get("args") or {})
    a.pop("scalar", None)
    return (nd["op_type"], tuple(_canon(pos, nd["inputs"])), nd["output"], json.dumps(a, sort_keys=True, default=str))


def cmp(rec_nodes, cap_nodes):
    ra, ca = _BR(rec_nodes, {}, {}, {}), _BR(cap_nodes, {}, {}, {})
    pa, pc = ra.pos(), ca.pos()
    res = "len ok" if len(rec_nodes) == len(cap_nodes) else f"LEN {len(rec_nodes)} vs {len(cap_nodes)}"
    nd, first = 0, None
    for i, (x, y) in enumerate(zip(rec_nodes, cap_nodes)):
        if ckey(x, pa) != ckey(y, pc):
            nd += 1
            if first is None:
                first = (i, ckey(x, pa)[:3], ckey(y, pc)[:3], ckey(x, pa)[3][:160], ckey(y, pc)[3][:160])
    try:
        sc, rt, zero = _diff(ca, ra)
        db = f"diff_block ok: {len(sc)} scalars, {len(rt)} rotations, {len(zero)} zero-cache inputs"
    except Exception as e:  # noqa: BLE001
        db = f"diff_block RAISES: {str(e)[:200]}"
    return res, nd, first, db


for b, br in sorted(r.records.get(0, {}).items()):
    d = os.path.join(work, "tok0", f"block_{b}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "graph.json"), "w") as f:
        json.dump({"version": 2, "nodes": br.nodes}, f)
_caps = {}
summary = collections.Counter()
for t, rec in sorted(r.records.items()):
    for b, br in sorted(rec.items()):
        src = g13 if b == 13 else gdir
        f = os.path.join(src, f"block_{b}", "graph.json")
        if not os.path.exists(f):
            print(f"token {t} block {b}: no {f}")
            continue
        if f not in _caps:
            _caps[f] = json.load(open(f))["nodes"]
        res, nd, first, db = cmp(br.nodes, _caps[f])
        summary[(t, res == "len ok" and nd == 0, db.startswith("diff_block ok"))] += 1
        print(f"token {t} block {b} vs {src}: {res}, {nd} differing canonical nodes; {db}; first={first}")
print("summary (token, identical, diff_block ok) -> blocks:", dict(summary))
import resource  # noqa: E402
print("done %.0fs, peak RSS %.0f MB" % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

# --- FULL16=1: precompute()'s structural check on all STEPS positions (diff_block vs the reference token + no
# dynamic step the probes kept constant), as PlannedRun.precompute/_steps do at load(), before out/ready
if os.environ.get("FULL16") == "1" and man is not None:
    import numpy as np  # noqa: E402
    rows = [np.zeros(h.dim)] * steps
    bad = 0
    for t in range(steps):
        recs = r.rec.token(t, rows[t], True)
        for b, rb in recs.items():
            try:
                s, k, z = _diff(r.ref[b], rb)
                for nm, (kind, v0) in r.ref_dyn[b].items():
                    (s if kind == "s" else k).pop(nm, None)
                if s or k:
                    raise RuntimeError(f"steps the probe tokens kept constant change: {(sorted(s) + sorted(k))[:6]}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FULL16 token {t} block {b}: {str(e)[:300]}", flush=True)
        del recs
        print(f"FULL16 token {t} ok so far (bad={bad}) rss peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
    print(f"FULL16: {steps} positions, {bad} block streams that would stop precompute()")

"""sanity2: Session.key_manifest (key_needs: the host-side GPU route plans) under bts.evalmod None / d27 / d27sb / bogus,
on the entry plan's 14 programs (compiled from the capture combo with the session Config)."""
import json, sys, time
from perseus import _pc
from perseus.serve import gpt2_decode as served
g, plan = sys.argv[1:3]
cfg = served._config(plan, "n32")
B = [f"block_{b}" for b in range(14)]
progs = _pc.Program.compile(g, plan, config=cfg, blocks=B, keep_going=True)
keys = _pc.KeySource.generate([], conj=True, layout="packed")
specs = {}
for em in (None, "d27", "d27sb", "bogus"):
    c = served._config(plan, "n32"); c["bts.evalmod"] = em
    t = time.time()
    try:
        s = json.loads(_pc.Session.key_manifest(c, keys, programs=[progs[b] for b in B], regs=256, pool=16, pt_rows=28,
                                                stc_cache=8, dyn_rot=[-i for i in range(-1, 32)]))
        specs[em] = sorted(json.dumps(k, sort_keys=True) for k in s["keys"])
        print(f"evalmod={em!r}: key_manifest ok in {time.time() - t:.0f}s, {len(s['keys'])} keys, contract {s['contract']}")
    except BaseException as e:  # noqa: BLE001  (pyo3 PanicException derives from BaseException)
        print(f"evalmod={em!r}: {type(e).__name__}: {str(e)[:300]}")
print("key lists equal across None/d27/d27sb:", len({json.dumps(v) for v in specs.values()}) == 1)

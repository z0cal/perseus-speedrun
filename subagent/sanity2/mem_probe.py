"""sanity2: onde o prepare() do servidor gasta memoria (RSS por etapa). Mesmo setup do prepare_entry.py, para
depois de N blocos do token 0. uso: mem_probe.py PLAN"""
import os
import sys
import threading
import time
import types

plan = sys.argv[1]


def rss():
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmRSS"):
                return int(ln.split()[1]) // 1024
    return -1


state = {"where": "import"}


def mon():
    while True:
        print(f"[mem] {time.strftime('%T')} rss={rss()} MB at {state['where']}", flush=True)
        time.sleep(1.0)


threading.Thread(target=mon, daemon=True).start()
from perseus.serve import gpt2_decode as served  # noqa: E402
from perseus.examples.gpt2 import weights as W  # noqa: E402
from perseus.impl import driver  # noqa: E402

_bw = W.block_weights


def bw(*a, **kw):
    state["where"] = f"block_weights({a[2]})"
    r = _bw(*a, **kw)
    state["where"] = f"after block_weights({a[2]})"
    print(f"[mem] block_weights({a[2]}) done rss={rss()} MB", flush=True)
    return r


W.block_weights = bw
if os.environ.get("LOWMEM") == "1":
    from perseus.examples.gpt2.model import Gpt2Model
    _obw = Gpt2Model.block_weights

    def _bwl(self, b):
        for k in list(self._bw):
            if k != b:
                del self._bw[k]
        return _obw(self, b)
    Gpt2Model.block_weights = _bwl
    _obr = driver.BlockRec.__init__

    def _br(self, nodes, pts, objs, metas):
        _obr(self, nodes, {k: None for k in pts}, objs, metas)
    driver.BlockRec.__init__ = _br
args = types.SimpleNamespace(plan=plan, chain="n32", layers=None, packing="cachemir_complex",
                             weights=os.environ["WEIGHTS_PATH"], key_layout="packed", pt_rows=28, stc_cache=8,
                             work_dir=os.environ.get("WORK", "/nonexistent"))
state["where"] = "handler"
h = served.Gpt2Handler(args)
print(f"[mem] handler rss={rss()} MB", flush=True)
r = h.run_
m = r.rec.model
_orig_token = r.rec.token


def tok(t, row, argmax=False):
    state["where"] = f"token {t}"
    out = _orig_token(t, row, argmax)
    print(f"[mem] token {t} recorded rss={rss()} MB", flush=True)
    return out


r.rec.token = tok
_ots = m.token_stages


def ts(x, pos):
    out = []
    for i, (lab, pre, fn) in enumerate(_ots(x, pos)):
        def wrap(x, fn=fn, i=i, lab=lab):
            state["where"] = f"stage {i} {lab}"
            y = fn(x)
            print(f"[mem] stage {i} {lab} done rss={rss()} MB", flush=True)
            return y
        out.append((lab, pre, wrap))
    return out


m.token_stages = ts
r.rec.values = False
import numpy as np  # noqa: E402
rows = [np.zeros(h.dim)] * 16
r._record(0, rows, True)
print(f"[mem] end rss={rss()} MB", flush=True)

"""sanity2: o que retem memoria no gemeo de CPU do servidor (tracemalloc, modelo de LAYERS camadas, token 0)."""
import gc, os, sys, tracemalloc, types
import numpy as np
plan = sys.argv[1]
from perseus.serve import gpt2_decode as served
from perseus.impl import driver
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
args = types.SimpleNamespace(plan=plan, chain="n32", layers=int(os.environ.get("LAYERS", "3")), packing="cachemir_complex",
                             weights=os.environ["WEIGHTS_PATH"], key_layout="packed", pt_rows=28, stc_cache=8, work_dir="/nonexistent")
h = served.Gpt2Handler(args)
r = h.run_
r.rec.values = False
tracemalloc.start(8)
rows = [np.zeros(h.dim)] * 16
r._record(0, rows, False)
gc.collect()
snap = tracemalloc.take_snapshot()
for st in snap.statistics("traceback")[:8]:
    print(f"{st.size/2**20:9.1f} MB  {st.count} blocks")
    for ln in st.traceback.format()[-12:]:
        print("      ", ln)

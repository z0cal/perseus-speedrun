"""Monkeypatch (capture only): the K/V(/Q) push WITHOUT its deliberate dense refresh (the planner places what it
needs). NOKV_LAYERS (env, comma list) limits it to some blocks (matched by call order mod 12)."""
import os
from perseus.impl import attention as A

_orig = A.cache_kv_push_packed_complex
_L = os.environ.get("NOKV_LAYERS", "")
LAYERS = None if not _L else {int(x) for x in _L.split(",")}
_count = [0]


def cache_kv_push_packed_complex(rt, kv, P_in, q=None, rope=None, pos=None):
    lay = _count[0] % 12
    _count[0] += 1
    if LAYERS is not None and lay not in LAYERS:
        return _orig(rt, kv, P_in, q, rope=rope, pos=pos)
    ops = rt.ops
    had = "bootstrap_real" in ops.__dict__
    saved = ops.__dict__.get("bootstrap_real")
    ops.bootstrap_real = lambda X, *a, **k: None
    try:
        return _orig(rt, kv, P_in, q, rope=rope, pos=pos)
    finally:
        if had:
            ops.bootstrap_real = saved
        else:
            del ops.bootstrap_real


def install():
    A.cache_kv_push_packed_complex = cache_kv_push_packed_complex

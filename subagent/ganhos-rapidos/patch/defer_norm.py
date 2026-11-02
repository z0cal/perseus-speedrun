"""Monkeypatch (capture only): the softmax's LAST reciprocal round is deferred past softmax_v.
probs = z * r is replaced by the pair (z, r); complex_softmax_v (SM_PERIODIC=1 branch) computes
Re(sum_lanes V * (z - i z)) through the token ladder and the tok0 mask on the z branch, then multiplies by r
(1/s is tH-periodic, constant over a head's t slots, so it commutes with the token rotsum and the slot mask).
Same values in exact arithmetic; the critical path loses the z*r product and the mask (2 levels).
DEFER_LAYERS (env, comma list of block indices; default all)."""
import os
from perseus.impl import attention as A
from perseus.impl.poly import goldschmidt_recip, im_cleanse, rotsum
from perseus.impl.layout import real_head_half_mask

_LAY = os.environ.get("DEFER_LAYERS", "")
LAYERS = None if not _LAY else {int(x) for x in _LAY.split(",")}


class Deferred:
    def __init__(self, z, r):
        self.z, self.r = z, r


def _cur_layer(rt):
    tag = getattr(rt, "_step_prefix", None)
    return tag


def _softmax_recip_deferred(rt, z, cfg, kc):
    ops = rt.ops
    fold = getattr(ops, "sm_fold", False) and not getattr(ops, "fused_sm_den", False)
    rounds = list(A._recip_rounds_ops(ops, cfg, kc))
    y = None
    for k, (alpha, beta, iters, scale, cheb) in enumerate(rounds):
        if k:
            z = ops.square(im_cleanse(ops, y))
        if fold:
            s, sh = A.head_reduce_sum(rt, z, s0_expected=2.0 / (alpha * scale), periodic=A.sm_periodic(rt, kc),
                                      seed=0.5 * beta * scale * scale, seed_site=A._seed_site(cfg, k))
            r = goldschmidt_recip(ops, s, alpha, beta, iters, scale, Dh=sh, cheb=cheb)
        else:
            s = A.head_reduce_sum(rt, z, s0_expected=2.0 / (alpha * scale), periodic=A.sm_periodic(rt, kc))
            r = goldschmidt_recip(ops, s, alpha, beta, iters, scale, cheb=cheb)
        if k == len(rounds) - 1:
            return Deferred(z, r)
        y = ops.mult(z, r)


_orig_softmax_thor = A.softmax_thor
_orig_csv = A.complex_softmax_v
_LAYER_IDX = {}


def softmax_thor(rt, scores, cfg, kc, folded=False):
    ops = rt.ops
    lay = _CLIPS.get((round(float(cfg.clip_lo), 9), round(float(cfg.clip_hi), 9)))
    if not ops.sm_den_recip or (LAYERS is not None and lay not in LAYERS):
        return _orig_softmax_thor(rt, scores, cfg, kc, folded)
    z = A._thor_exp(rt, scores, cfg, kc, folded)
    return _softmax_recip_deferred(rt, z, cfg, kc)


def complex_softmax_v(rt, kv, probs):
    if not isinstance(probs, Deferred):
        return _orig_csv(rt, kv, probs)
    ops, d = rt.ops, rt.dims
    per = A.sm_periodic(rt, kv.k_count)
    if per != 1 or A._general(rt):
        raise NotImplementedError("defer_norm: SM_PERIODIC=1 branch only")
    n_pairs = d.d_head_real // 2
    vs = ops.copy(kv.v_buckets[0])
    for b in kv.v_buckets[1:n_pairs]:
        ops.inplace_add(vs, b)
    z = probs.z
    res = ops.mult(vs, ops.sub(z, ops.mult_i(z)))
    res = rotsum(ops, res, 1, d.t)
    res = im_cleanse(ops, res)
    res = rt.mult_mask(res, "tok0.h", lambda: real_head_half_mask(d))
    return ops.mult(res, probs.r)


_CLIPS = {}


def install():
    A.softmax_thor = softmax_thor
    A.complex_softmax_v = complex_softmax_v
    cfgp = os.environ.get("DEFER_CONFIGS")
    if cfgp:
        import json
        c = json.load(open(cfgp))
        for l in range(12):
            s = c["softmax"][f"transformer.h.{l}.attn"]
            _CLIPS[(round(float(s["clip_lo"]), 9), round(float(s["clip_hi"]), 9))] = l

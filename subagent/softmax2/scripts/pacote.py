"""Per centred layer, candidate packages (exp seed (d, k1) by the mixed-criterion LP + Goldschmidt counts), judged by
softmax_ref per row kind, out-of-range rows, round bands and exp-refresh noise. usage: pacote.py [LAYERS]"""
import dataclasses, json, math, sys
import numpy as np
import sm2, noise, oor
from perseus.impl import smcheb
sys.path.insert(0, "/home/user/perseus-speedrun/scripts")
from gelu_bsgs_count import bsgs, trim

KINDS = ("uniform", "cluster", "real", "flat", "top", "bottom")


def exp_cost(coeffs, k1):
    c = trim(coeffs, 1e-6)
    a, b = bsgs(c)
    return a + k1, b + k1, len(c) - 1


def gs_depth(ks):
    return ks[0] + 2 + sum(k + 3 for k in ks[1:])


def with_gs(cfg, ks, tg=None):
    return dataclasses.replace(cfg, recip_iters_cheb=tuple(ks), recip_targets=tuple(tg) if tg else cfg.recip_targets)


def band_use(cfg, j):
    """min/max of the scaled denominator per round over all row kinds, against the round's band."""
    rng = np.random.default_rng(9)
    tr = {}
    for kind in KINDS:
        for kc in (1, 2, 4, 8, 16):
            u = sm2.rows(j, kc, 500, rng, kind, True)
            t = {}
            noise.run(cfg, u, kc, track=t)
            for k, v in t.items():
                lo = min(x[0] for x in v); hi = max(x[1] for x in v)
                b = noise.bands(cfg, kc)[k]
                r = tr.setdefault(k, [np.inf, -np.inf, np.inf, -np.inf])
                r[0] = min(r[0], lo / b[0]); r[1] = max(r[1], hi / b[1])
    return " ".join(f"r{k}: min/piso {v[0]:.2f} max/topo {v[1]:.3f}" for k, v in sorted(tr.items()))


def show(name, j, cfg, k1, ks, rec=None):
    ev = sm2.evaluate(j, cfg, N=3000, seed=11, kinds=KINDS)
    a = ev["ALL"]
    per = " ".join(f"{k}:{ev[k]['maxdp']:.1e}/{ev[k]['dout']:.1e}" for k in ("real", "flat", "uniform"))
    lv, pr, dd = exp_cost(cfg.cheb_coeffs, k1)
    ratio = "" if rec is None else f" | razao maxdp {a['maxdp']/rec['maxdp']:.2f} dout {a['dout']/rec['dout']:.2f} tv {a['tv']/rec['tv']:.2f}"
    print(f"  {name:30s} exp (niv {lv}, prod {pr}, grau efetivo {dd}) GS {list(ks)} (niv {gs_depth(ks)}) total {lv + gs_depth(ks)}"
          f" | ALL maxdp {a['maxdp']:.2e} tv {a['tv']:.2e} dout {a['dout']:.2e} fac {a['fac']:.1e} shape {a['shape']:.1e}"
          f" NaN {a['nan']}{ratio}", flush=True)
    print(f"  {'':30s} por tipo (maxdp/dout) {per}")
    c = dict(j); c.update(cheb_coeffs=list(cfg.cheb_coeffs))
    print(f"  {'':30s} acima de U: {oor.above(c, cfg, xs=(0.5, 1, 1.5, 2, 3))}")
    print(f"  {'':30s} abaixo de lo: {oor.below(c, cfg, xs=(1, 2, 5, 10, 20))}")
    print(f"  {'':30s} bandas: {band_use(cfg, j)}")
    am = noise.amp(j, cfg, N=600)
    print(f"  {'':30s} ruido exp (rms,max) x: " + " ".join(f"{k}:{v[0]:.0f},{v[1]:.0f}" for k, v in am.items()), flush=True)
    return a


def main():
    layers = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    for L in layers:
        j, cfg = sm2.layer(L)
        ks0 = [int(x) for x in cfg.recip_iters_cheb]
        print(f"== camada {L} (W {j['clip_hi'] - j['clip_lo']:.1f}, d1 {j['delta1']}, d2 {j['delta2']}, GS {ks0}, alvos {j['recip_targets']})")
        rec = show("recorde", j, cfg, cfg.log2delta1, ks0)
        k1r = cfg.log2delta1
        # Goldschmidt alternatives with the same or +1 total in the last (slack) round
        R = cfg.log2delta2
        alts = []
        for tgt in (1e-4, 1e-5):
            ks, tg = smcheb.design(cfg.init_alpha, cfg.init_beta, R, tgt_last=tgt, kcs=tuple(range(1, 17)))
            if sum(ks) == sum(ks0) and list(ks) != ks0:
                alts.append((ks, tg)); break
        last1 = ks0[:-1] + [ks0[-1] + 1]
        alts.append((last1, None))
        for ks, tg in alts:
            show(f"recorde + GS {list(ks)}", j, with_gs(cfg, ks, tg), k1r, ks, rec)
        for d, k1 in ((14, 0), (14, 1)):
            if k1 >= k1r: continue
            co, eps = sm2.fit_lp(j, cfg, d, k1)
            c2 = sm2.with_seed(j, cfg, co, k1)
            show(f"semente ({d},{k1})", j, c2, k1, ks0, rec)
            for ks, tg in alts:
                show(f"semente ({d},{k1}) + GS {list(ks)}", j, with_gs(c2, ks, tg), k1, ks, rec)


if __name__ == "__main__":
    main()

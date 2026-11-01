"""Out-of-range rows: one or several keys at U + x (above the clip) or at lo - x (below), through softmax_ref.
Reports the max error and whether the result is finite, per layer config. Also round-0 Goldschmidt past the band top."""
import math, sys
import numpy as np
import sm2, h3
from perseus.impl.config import parse_softmax
from perseus.impl import smcheb
from perseus.impl.poly import goldschmidt_recip
from perseus.examples.gpt2 import ref as R


def above(c, cfg, xs=(0.5, 1, 2, 3, 4, 6, 8), kcs=(2, 4, 16), nup=(1, 2, 4)):
    out = []
    for x in xs:
        worst = 0.0
        for kc in kcs:
            for n in nup:
                if n >= kc: continue
                u = np.full((1, kc), -3.0); u[0, 0] = 0.0
                u[0, 1:1 + n] = c["clip_hi"] + x
                p = sm2.sm(cfg, u, kc); pe = sm2.exact(u)
                e = np.abs(p - pe).max() if np.all(np.isfinite(p)) else np.inf
                worst = max(worst, e)
        out.append(f"+{x:g}:{worst:.1e}")
    return " ".join(out)


def below(c, cfg, xs=(1, 2, 5, 10, 20, 40), kcs=(2, 16)):
    out = []
    for x in xs:
        worst = 0.0
        for kc in kcs:
            u = np.full((1, kc), c["clip_lo"] - x); u[0, 0] = 0.0
            p = sm2.sm(cfg, u, kc); pe = sm2.exact(u)
            e = np.abs(p - pe).max() if np.all(np.isfinite(p)) else np.inf
            worst = max(worst, e)
        out.append(f"-{x:g}:{worst:.1e}")
    return " ".join(out)


def gs_top(cfg):
    a, b, it, sc, ch = smcheb.rounds(cfg, 16)[0]
    out = []
    for D in (1.05, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5):
        r = goldschmidt_recip(R._ops, np.array([D]), a, b, it, sc, Dh=0.5 * b * sc * sc * np.array([D]), cheb=ch)
        out.append(f"D={D}:{abs(np.real(r[0]) * D - 1):.1e}")
    return " ".join(out)


if __name__ == "__main__":
    j, cfg = sm2.layer(4)
    us, ul = 18.57, -96.88
    variants = {"recorde L4": dict(j),
                "L4 d2 4 m1.0 d1 4 g12": h3.build(j, us, ul, margin=1.0, delta2=4, delta1=4, degree=12),
                "L4 d2 4 m1.1 d1 4 g12": h3.build(j, us, ul, margin=1.1, delta2=4, delta1=4, degree=12),
                "L4 d2 4 m1.25 d1 8 (tau5)": h3.build(j, us, ul, tau=5.0)}
    for L in (0, 10):
        variants[f"recorde L{L}"] = dict(sm2.layer(L)[0])
    for name, c in variants.items():
        cfg = parse_softmax(c)
        print(f"{name:28s} acima de U: {above(c, cfg)}")
        print(f"{'':28s} abaixo de lo: {below(c, cfg)}")
        print(f"{'':28s} GS rodada 0 acima do topo: {gs_top(cfg)}", flush=True)

"""smrel sobre a exp do recorde 7: a série servida tem grau 6 (profundidade BSGS 3) e n_squarings s quadraturas.
Tirar uma quadratura (smrel r = 1) pede a série de exp(2 c t) no mesmo t em [-1, 1], com erro relativo final igual:
2^(s-1) eps_d(2c) <= 2^s eps_6(c). eps_d = erro relativo máximo do interpolante de Chebyshev (quase minimax) de grau d.
Profundidade do BSGS do Perseus (relatório softmax2, coeficientes genéricos): grau 5-6 -> 3 níveis, 7-15 -> 4, 16-31 -> 5.
Uso: python3 -I smrel_r7.py UPSTREAM_CLONE"""
import json, sys
import numpy as np
from numpy.polynomial import chebyshev as Ch
C = json.load(open(sys.argv[1] + "/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab/configs.json"))["softmax"]
t = np.cos(np.linspace(0, np.pi, 4001))


def eps(c, d):
    f = lambda x: np.exp(c * x)  # noqa: E731
    p = Ch.Chebyshev.interpolate(f, d)
    return float(np.max(np.abs(p(t) / f(t) - 1)))


def depth(d):
    return 3 if d <= 6 else (4 if d <= 15 else 5)


tot_now, tot_new = 0, 0
for L in range(12):
    c = C[f"transformer.h.{L}.attn"]
    s, cb, deg = int(c["n_squarings"]), float(c["cheb_b"]), len(c["cheb_coeffs"]) - 1
    e_now = 2 ** s * eps(cb, deg)
    need = next(d for d in range(2, 40) if 2 ** (s - 1) * eps(2 * cb, d) <= e_now)
    lv_now, lv_new = depth(deg) + s, depth(need) + s - 1
    tot_now += lv_now; tot_new += lv_new
    print(f"camada {L:2d}: c {cb:.3f}, s {s}, grau {deg}: erro relativo final {e_now:.1e}; sem 1 quadratura precisa de grau "
          f"{need:2d} -> níveis exp+quadraturas {lv_now} -> {lv_new} ({lv_new - lv_now:+d}; depois do SB#1: {s} -> {s - 1})")
print(f"total: {tot_now} -> {tot_new} níveis/token na exp ({tot_new - tot_now:+d})")

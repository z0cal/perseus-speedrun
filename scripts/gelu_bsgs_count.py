"""Product/depth count of Perseus's eval_chebyshev_bsgs (re-implemented from python/perseus/impl/poly.py at
upstream d97a6f7: babies T_2..T_{m-1}, giants T_m..T_{2^(L-1)}, IMPL_DEAD_OPS pruning via _cheb_needed, recursive
split q(T) T_k + r(T), leaves as scalar sums), for the GELU series of a config, against the EVEN-in-z form
(T_2j(x) = T_j(z), z = 2x^2 - 1: one square, then a series of half the degree; Trefethen ATAP ch. 3).

Depth model (same as subagent/depth_trace.py): ct*ct product / square = max + 1, leaf scalar sum = max + 1, adds free,
the affine map is free on [-1, 1] (alpha = 1, beta = 0: the GELU configs use cheb_a = -1, cheb_b = 1).
Pure Python, no Perseus import. usage: python -I gelu_bsgs_count.py CONFIGS_JSON
"""
import json
import sys


def trim(c, eps):
    """poly._cheb_trim: coefficients <= eps * max|c| set to 0, trailing zeros cut."""
    c = [float(x) for x in c]
    tol = eps * max(abs(x) for x in c)
    c = [0.0 if abs(x) <= tol else x for x in c]
    n = len(c) - 1
    while n > 0 and c[n] == 0.0:
        n -= 1
    return c[:n + 1]


def split(c, k):
    """poly._cheb_split: sum c_i T_i = q(T) T_k + r(T)."""
    d = len(c) - 1
    q = [0.0] * (d - k + 1)
    r = list(c[:k])
    q[0] = c[k]
    for j in range(1, d - k + 1):
        q[j] = 2.0 * c[k + j]
        r[k - j] -= c[k + j]
    return q, r


def strip(c):
    d = len(c) - 1
    while d > 0 and c[d] == 0.0:
        d -= 1
    return c[:d + 1]


def needed(c, m):
    """poly._cheb_needed: the T_i the recursion reads, closed under their construction."""
    used = set()

    def rec(c):
        c = strip(c)
        d = len(c) - 1
        if d < m:
            used.update(i for i in range(1, len(c)) if c[i] != 0.0)
            return
        k = 1 << (d.bit_length() - 1)
        used.add(k)
        q, r = split(c, k)
        rec(q)
        rec(r)

    rec(list(c))
    need, todo = set(), sorted(used, reverse=True)
    while todo:
        i = todo.pop()
        if i in need or i <= 1:
            continue
        need.add(i)
        todo += [i // 2] if (i >= m or i % 2 == 0) else [i // 2 + 1, i // 2]
    return need


def bsgs(c, dead=True, d0=0, m=None):
    """(depth, ct*ct products) of eval_chebyshev_bsgs; d0 = input depth; m = baby size (default = Perseus's)."""
    n = len(c) - 1
    L = n.bit_length()
    m = (1 << ((L + 1) // 2)) if m is None else m
    need = needed(c, m) if dead else None
    T = {1: d0}
    nmul = 0
    for i in range(2, m):
        if need is not None and i not in need:
            continue
        T[i] = (T[i // 2] if i % 2 == 0 else max(T[i // 2 + 1], T[i // 2])) + 1
        nmul += 1
    k = m
    while k <= n:
        if need is None or k in need:
            T[k] = T[k // 2] + 1
            nmul += 1
        k *= 2

    def leaf(c):
        nz = [i for i in range(1, len(c)) if c[i] != 0.0]
        return (max(T[i] for i in nz) + 1) if nz else d0

    def rec(c):
        nonlocal nmul
        c = strip(c)
        d = len(c) - 1
        if d < m:
            return leaf(c)
        k = 1 << (d.bit_length() - 1)
        q, r = split(c, k)
        dq = rec(q)
        nmul += 1
        dp = max(dq, T[k]) + 1
        return max(dp, rec(r))

    return rec(c), nmul


def frontier(pts):
    """Pareto points (depth, products), sorted by depth."""
    out = []
    for d, n in sorted(set(pts)):
        if not out or n < out[-1][1]:
            out.append((d, n))
    return out


def main():
    cfg = json.load(open(sys.argv[1]))
    tot = {"atual": 0, "z_mesmo_nivel": 0}
    print("camada | grau | atual (niveis, ct*ct) | sem dead_ops | x, outros m | z = 2x^2-1 (grau_z): (niveis, ct*ct)")
    for k, g in cfg["softgelu"].items():
        c = trim(g["cheb_coeffs"], g.get("cheb_eps", 1e-6))
        odd = max((abs(x) for i, x in enumerate(c) if i % 2 and i >= 3), default=0.0)
        ce = [c[2 * j] for j in range((len(c) - 1) // 2 + 1)]
        cur = bsgs(c)
        fx = frontier(bsgs(c, True, 0, 1 << b) for b in range(1, 9) if (1 << b) <= 2 * len(c))
        fz = frontier((d, n + 1) for d, n in (bsgs(ce, True, 1, 1 << b) for b in range(1, 9) if (1 << b) <= 2 * len(ce)))
        same = min(n for d, n in fz if d <= cur[0]) if any(d <= cur[0] for d, _ in fz) else None
        tot["atual"] += cur[1]
        tot["z_mesmo_nivel"] += same if same is not None else cur[1]
        name = k.split(".")[2] if k.count(".") >= 2 else k
        print(f"{name:>6} | {len(c) - 1:4d} | {cur} | {bsgs(c, False)} | {fx} | ({len(ce) - 1}) {fz}"
              + ("" if odd == 0.0 else f"  ATENCAO: c_impar>=3 nao nulo ({odd:.1e})"))
    print(f"total ct*ct por token (1 ciphertext por camada): atual {tot['atual']}, "
          f"par em z sem perder nivel {tot['z_mesmo_nivel']}")


if __name__ == "__main__":
    main()

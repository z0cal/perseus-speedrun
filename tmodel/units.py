"""Work units of CKKS device ops, per ciphertext, as three features: NTT limbs, basis-conversion MAC limbs / 16, and
elementwise limb passes. A time model is t = tau * ntt + kappa * conv + eps * ew (+ a per-launch constant).

Chains: n32 (54 Q primes, dnum 6 -> digits of 9, 9 special primes), n32a4 (51 Q, dnum 4 -> digits of 13, 13 special).
`n` is a limb count (Q primes held). Composite rescales drop 2 limbs."""
import json
import math

CHAINS = {"n32": dict(q=54, a=9, alpha=9), "n32a4": dict(q=51, a=13, alpha=13)}


def z():
    return [0.0, 0.0, 0.0]


def add(u, v, s=1.0):
    for i in range(3):
        u[i] += s * v[i]
    return u


def ks(n, ch):
    """One full key switch of one ciphertext at n limbs (ModUp of c1, key inner product, ModDown of both polys)."""
    a, al = CHAINS[ch]["a"], CHAINS[ch]["alpha"]
    b = math.ceil(n / a)
    ntt = n + b * (n + al) + 2 * al + 2 * n
    conv = (b * a * (n + al - a) + 2 * al * n) / 16.0
    ew = 3 * b * (n + al) + 4 * n
    return [ntt, conv, ew]


def ks_hoisted_extra(n, ch):
    """One more rotation on an already ModUp'ed c1, accumulated in the extended basis (no ModDown of its own)."""
    a, al = CHAINS[ch]["a"], CHAINS[ch]["alpha"]
    b = math.ceil(n / a)
    return [0.0, 0.0, 3 * b * (n + al)]


def moddown(n, ch):
    al = CHAINS[ch]["alpha"]
    return [2 * al + 2 * n, 2 * al * n / 16.0, 2 * (n + al)]


def modup(n, ch):
    a, al = CHAINS[ch]["a"], CHAINS[ch]["alpha"]
    b = math.ceil(n / a)
    return [n + b * (n + al), b * a * (n + al - a) / 16.0, 0.0]


def rescale(n):
    """Composite rescale (2 limbs) of both polys."""
    return [2 * n + 4, 0.0, 2 * n]


def lt_stage(n, bstep, g, npts, ch, giant_identity=False, offset=False):
    """Hoisted BSGS stage: ModUp once, b-1 hoisted baby rotations, npts pt products in Q u P summed per giant step,
    one ModDown per giant step, g-1 giant rotations (Horner) unless the giant step is the identity."""
    al = CHAINS[ch]["alpha"]
    u = modup(n, ch)
    for _ in range(bstep - 1):
        add(u, ks_hoisted_extra(n, ch))
    add(u, [0.0, 0.0, npts * 2 * (n + al)])
    for _ in range(g):
        add(u, moddown(n, ch))
    if not giant_identity:
        for _ in range(g - 1):
            add(u, ks(n, ch))
    if offset:
        add(u, ks(n, ch))
    add(u, rescale(n))
    return u


def evalmod(oplist, ch):
    """EvalMod from an emcost op list ([kind, n, pairs]); relins are full key switches."""
    u = z()
    for k, n, p in oplist:
        if k in ("mult", "multacc"):
            add(u, [0.0, 0.0, 4 * n * p])
            add(u, ks(n, ch))
        elif k == "rescale":
            add(u, rescale(n))
        elif k in ("add", "mulconst", "copy"):
            add(u, [0.0, 0.0, 2 * n])
        elif k == "addconst":
            add(u, [0.0, 0.0, n])
        elif k == "wsum":
            add(u, [0.0, 0.0, 2 * n * p])
    return u


def raise_(n_top):
    """ModRaise of both polys from q0 (2 limbs) to n_top limbs: INTT 2 limbs, NTT n_top limbs, per poly."""
    return [2 * (n_top + 2), 0.0, 2 * n_top]


def load_em(path):
    return {json.loads(l)["name"]: json.loads(l) for l in open(path)}

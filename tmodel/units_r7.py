"""units.py's work units for the runtime of records 5-7 (perseus speedrun e805f8d), on n32a4 / n32sp (51 Q primes, 13
special, digits [13, 13, 13, 12]):
- a relinearization followed by the rescale of its result runs FUSED (pc-gpu `RelinRescale`, PC_EM_FUSE_RESCALE /
  compile rr): the ModDown converts from the specials and the two dropped tops straight to the n - 2 remaining limbs,
  and there is no separate rescale;
- in EvalMod a fused one at limb top 16..46 (even) uses the LEVEL relinearization key (PC_EM_LEVEL_KEYS: pc-core
  `Chain::level_relin_layout` with cap 14): P' = the idle Q primes above the level + P (<= 14), fewer, larger digits.
Everything else is units.py's (same features: NTT limbs, conversion MACs / 16, elementwise limbs)."""
import math

import units as U

NQ, KP, CAP = 51, 13, 14
DIGIT_STARTS = (0, 13, 26, 39)
LEVEL_TOPS = set(range(16, 47, 2))      # pc-py session: the level keys it carves


def level_layout(top):
    """(digits, digit size, |P'|) of the level relinearization key at limb top `top`, or None (pc-core
    Chain::level_relin_layout)."""
    n = top + 1
    if top not in LEVEL_TOPS or n >= NQ or top < 2:
        return None
    pmax = min(KP + (NQ - n), CAP)
    d = math.ceil(n / pmax)
    cur = sum(1 for f in DIGIT_STARTS if f <= top)
    if d >= cur:
        return None
    a = math.ceil(n / d)
    e = min(max(a - KP, 0), NQ - n)
    return d, a, e + KP


def ks_general(n, b, a, al, fused_drop=0):
    """One key switch at n limbs with b digits of a limbs over al specials; fused_drop > 0: the ModDown fused with a
    rescale dropping that many tops (converts from al + drop rows to n - drop)."""
    out = n - fused_drop
    src = al + fused_drop
    ntt = n + b * (n + al) + 2 * src + 2 * out
    conv = (b * a * (n + al - a) + 2 * src * out) / 16.0
    ew = 3 * b * (n + al) + 4 * n
    return [ntt, conv, ew]


def relin(n, fused=False, level_key=False, drop=2):
    """Tensor (4 n) + key switch of d2; fused: the rescale (`drop` tops: 2, or 1 on an alternating region's
    one-prime level) inside the ModDown; level_key: the level key's digits and P' where pc-gpu has one."""
    u = [0.0, 0.0, 4.0 * n]
    lay = level_layout(n - 1) if (fused and level_key) else None
    if lay is not None:
        b, a, al = lay
    else:
        b, a, al = math.ceil(n / 13), 13, KP
    U.add(u, ks_general(n, b, a, al, fused_drop=drop if fused else 0))
    return u


def evalmod(oplist, level_keys=True):
    """An em_prog op list (pc-gpu EvalModPlan::program: fused, rewritten, relin-rescales scheduled; 'multrs' /
    'multaccrs' = the fused RelinRescale pairs)."""
    u = U.z()
    for k, n, p in oplist:
        if k in ("mult", "multacc"):
            U.add(u, relin(n))
            if k == "multacc":
                U.add(u, [0.0, 0.0, 4.0 * n * (p - 1)])
        elif k in ("multrs", "multaccrs"):
            U.add(u, relin(n, fused=True, level_key=level_keys))
            if k == "multaccrs":
                U.add(u, [0.0, 0.0, 4.0 * n * (p - 1)])
        elif k == "rescale":
            U.add(u, U.rescale(n))
        elif k in ("add", "mulconst", "copy"):
            U.add(u, [0.0, 0.0, 2 * n])
        elif k == "addconst":
            U.add(u, [0.0, 0.0, n])
        elif k == "wsum":
            U.add(u, [0.0, 0.0, 2 * n * p])
    return u

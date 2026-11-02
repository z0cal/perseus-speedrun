import json, sys, os
T='/home/user/perseus-speedrun/tmodel'
sys.path.insert(0, T)
import units as U
_l = U.load_em
def lw(path):
    d = _l(path); ev = path.replace('/n32a4_t', '/even_n32a4_t')
    if os.path.exists(ev): d.update(_l(ev))
    return d
U.load_em = lw
import tokplan as TP, predict as P
ms = P.ms
for route, real, d0 in [('stcf', True, 0), ('stcf', False, 0), ('stcf', True, 1), ('512', False, 1), ('512', False, 5)]:
    s = dict(route=route, real=real, drop=d0, lout=26)
    base = ms(TP.site_units(s, 'shipped', 'shipped'))
    s1 = dict(s, drop=d0 + 1)
    drop_only = ms(TP.site_units(s1, 'shipped', 'shipped')) - base   # same series, one more drop (landing would move!)
    if route == 'stcf':
        ev = ms(TP.site_units(s1, 'Even_z11_K16', 'shipped')) - base
        tab_only = ms(TP.site_units(s, 'Even_z11_K16', 'shipped')) - base
    else:
        ev = ms(TP.site_units(s1, 'shipped', 'Even_z13_K24')) - base
        tab_only = ms(TP.site_units(s, 'shipped', 'Even_z13_K24')) - base
    print(f"{route:5s} real={real!s:5s} drop {d0}: base {base:6.2f} ms | even+1drop {ev:+.3f} | shipped+1drop (raise/CtS/EvalMod lower) {drop_only:+.3f} | even same drop {tab_only:+.3f}")

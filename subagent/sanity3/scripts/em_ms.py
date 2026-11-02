import json, sys
sys.path.insert(0, '/home/user/perseus-speedrun/tmodel')
import units as U
C = json.load(open('/home/user/perseus-speedrun/tmodel/calib.json'))
ms = lambda u: C['tau'] * u[0] + C['kappa'] * u[1] + C['eps'] * u[2]
def L(nl, t):
    d = {}
    for f in (f'nl{nl}_ship_t{t}.jsonl', f'nl{nl}_mine_t{t}.jsonl'):
        for l in open(f):
            x = json.loads(l); d[x['name']] = x
    return d
def em(nl, t, n):
    return ms(U.evalmod(L(nl, t)[n]['oplist'], 'n32a4'))
print("EvalMod ms per chain (calib.json), census convention (nl=2 lists, tmodel tops) vs runtime metadata")
print("DENSE  tmodel: shipped t42 nl2 %.3f | even z11 t40 nl2 %.3f | d27 t40 nl2 %.3f | z13 t40 nl2 %.3f" % (em(2,42,'Shipped_K16'), em(2,40,'Even_z11_K16'), em(2,40,'D27_K16'), em(2,40,'Even_z13_K16')))
print("DENSE  real  : shipped t42 nl1 %.3f | even z11 t40 nl1 %.3f | d27 t40 nl1 %.3f | z13 t40 nl1 %.3f" % (em(1,42,'Shipped_K16'), em(1,40,'Even_z11_K16'), em(1,40,'D27_K16'), em(1,40,'Even_z13_K16')))
print("SPARSEB tmodel: shipped t46 nl2 %.3f | even z13 t44 nl2 %.3f | d27 t44 nl2 %.3f" % (em(2,46,'Shipped_K24'), em(2,44,'Even_z13_K24'), em(2,44,'D27_K24')))
print("SPARSEB real  : shipped t48 nl2 %.3f | even z13 t46 nl2 %.3f | d27 t46 nl2 %.3f" % (em(2,48,'Shipped_K24'), em(2,46,'Even_z13_K24'), em(2,46,'D27_K24')))

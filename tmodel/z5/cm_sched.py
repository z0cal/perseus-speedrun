"""Cronogramas do CutMax em float64: passos de Newton pelo ajuste do upstream, nitidez (top/soma|x|) e argmax."""
import sys as _s, unittest.mock as _m
for _n in ("torch","torch.nn","torch.nn.functional","transformers"): _s.modules[_n]=_m.MagicMock()
import json, math, os, sys, itertools
import numpy as np
R7=sys.argv[1]; SP=sys.argv[2]
sys.path.insert(0,f"{R7}/python"); sys.path.insert(0,f"{R7}/speedrun/harness")
from perseus.calibrate.approximations import cutmax as CM
from perseus.examples.gpt2 import ref as R
rows=np.load(f"{SP}/rows.npy").astype(np.float64)
ex=f"{SP}/rows_cm.npy"
if os.path.exists(ex):
    rows_cal=np.concatenate([rows,np.load(ex).astype(np.float64)-np.load(ex).astype(np.float64).mean(axis=1,keepdims=True)])
else: rows_cal=rows
base=json.load(open(f"{R7}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab/configs.json"))["cutmax"]
print("recorde: newton",[len(s) for s in base["newton"]],"sum",len(base["sum_newton"]))
def lv(p): return math.ceil(math.log2(p+1))
def run(ps,cs,tol=5e-3,sum_tol=1e-5,label=""):
    sec=dict(entry_scale=base["entry_scale"],p=list(ps),c=list(cs),m=[1.0]*len(ps))
    # m fixa só a escala; só a razão da banda importa para os passos
    s=CM.fit_scaled([rows_cal],sec,tol=tol,sum_tol=sum_tol)
    nw=[len(x) for x in s["newton"]]
    n=rows.shape[1]
    out=[];ok=0
    for r in rows:
        top=int(np.argmax(r)); x=r*base["entry_scale"]
        for i,p in enumerate(ps):
            d=x-x.sum()/n
            S2=np.array([np.sum(d*d)])
            f=R.inv_sqrt_newton_scaled(R._ops,S2*s["x_scale"][i],s["newton"][i],0.5*math.sqrt(n*s["x_scale"][i])/(cs[i]*1.0))[0]
            y=2.0*(d*f+0.5/1.0)
            x=R.pow_odd(R._ops,y,p)
        a=np.abs(x); out.append(a[top]/a.sum()); ok+=int(np.argmax(x)==top)
    o=np.array(out)
    print(f"{label or str(ps):26s} c={list(cs)} tol={tol:g} newton={nw}+{len(s['sum_newton'])} niveis_pow={sum(lv(p) for p in ps)} prodP={int(np.prod(ps))} argmax={100*ok/len(rows):.2f}% top/soma med={np.median(o):.4f} p10={np.percentile(o,10):.4f} p01={np.percentile(o,1):.4f} min={o.min():.4f}",flush=True)
c0=7.0;c=10.0
run([9,3,3,3,7],[7,10,10,10,10],label="recorde")
run([9,3,3,3,7],[7,10,10,10,10],tol=2e-2,label="recorde tol2e-2")
run([9,3,3,3,7],[7,10,10,10,10],tol=1e-1,label="recorde tol1e-1")
for ps,cs in [([9,3,7,7],[7,10,10,10]),([15,3,3,7],[7,10,10,10]),([15,7,7],[7,10,10]),([15,7,15],[7,10,10]),([9,7,15],[7,10,10]),([15,15,7],[7,10,10]),([31,7,7],[7,10,10]),([15,15],[7,10]),([31,15],[7,10]),([31,31],[7,10])]:
    run(ps,cs); run(ps,cs,tol=5e-2)

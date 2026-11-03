"""CutMax em float64 sobre logits exatos do GPT-2 (janelas OWT, posições 8-15; nenhum prompt de dev): nitidez e argmax
a cada iteração. O harness compara argmax(z) com argmax(logits)."""
import json, math, os, sys, time
os.environ.setdefault("SM_CHEB_GS","1")
import numpy as np
R7=sys.argv[1]; N=int(sys.argv[2])
sys.path.insert(0,f"{R7}/python"); sys.path.insert(0,f"{R7}/speedrun/harness")
from gpt2_ref import GPT2
from perseus.examples.gpt2 import ref as R
from perseus.impl import config as C
import glob
w=sorted(glob.glob(os.path.expanduser("~/pesquisa/perseus-speedrun/work/hf/hub/models--openai-community--gpt2/snapshots/*/model.safetensors")))[0]
g=GPT2(w)
pool=np.load(os.path.expanduser("~/pesquisa/perseus-speedrun/work/hf/perseus/pools/openwebtext_gpt2_2000000.npy"),mmap_mode="r")
cfg=C.parse_cutmax(json.load(open(f"{R7}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab/configs.json"))["cutmax"])
print("iters",len(cfg.iters),"p",[it.p for it in cfg.iters],"entry",cfg.entry_scale)
rows=[]
t=time.time()
for o in range(3_000_000%len(pool), 3_000_000%len(pool)+6101*N, 6101):
    ids=[int(i) for i in pool[o%(len(pool)-16):o%(len(pool)-16)+16]]
    lg=g.logits(g.embed(ids))
    rows.append(np.asarray(lg[8:16],dtype=np.float64))
rows=np.concatenate(rows); print("rows",rows.shape,"t",round(time.time()-t))
rows=rows-rows.mean(axis=1,keepdims=True)   # lm_centre desloca por constante
np.save(sys.argv[3],rows.astype(np.float32))
def stats(x,top):
    a=np.abs(x); s=a.sum(); m=a[top]
    o=np.delete(a,top)
    return m, o.max(), m/s
agg=[]
for r in rows:
    top=int(np.argmax(r))
    x=r*cfg.entry_scale
    n=x.shape[0]; rec=[]
    rec.append((int(np.argmax(x))==top,)+stats(x,top))
    for it in cfg.iters:
        d=x-x.sum()/n
        S2=np.array([np.sum(d*d)])
        f=R.inv_sqrt_newton_scaled(R._ops,S2*it.x_scale,it.newton,0.5*math.sqrt(n*it.x_scale)/(it.c*it.m))[0]
        y=2.0*(d*f+0.5/it.m)
        x=R.pow_odd(R._ops,y,it.p)
        rec.append((int(np.argmax(x))==top,)+stats(x,top))
    agg.append(rec)
A=np.array(agg,dtype=np.float64)
print("etapa  argmax%  top(mediana)  2o(mediana)  top/soma|x|: mediana p10 p01 min")
for k in range(A.shape[1]):
    print(k, f"{100*A[:,k,0].mean():7.2f}", f"{np.median(A[:,k,1]):.3e}", f"{np.median(A[:,k,2]):.3e}", " ".join(f"{np.percentile(A[:,k,3],q):.4f}" for q in (50,10,1,0)))
s=np.sort(rows,axis=1); gap=s[:,-1]-s[:,-2]
print("margem top1-top2 nos logits: mediana %.3f p10 %.3f p01 %.4f min %.5f; sigma logits mediana %.2f"%(np.median(gap),np.percentile(gap,10),np.percentile(gap,1),gap.min(),np.median(rows.std(axis=1))))

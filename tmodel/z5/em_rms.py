"""EvalMod: o que importa no slot é o erro quadrático médio sobre a distribuição real de I (soma de h+1 = 33 uniformes,
sigma 1,66), não o máximo em |I| <= K-1. Ajuste por mínimos quadrados ponderados no erro FINAL (Gauss-Newton com LM)."""
import math,sys
import numpy as np
from fractions import Fraction
sys.path.insert(0,"/home/zocal/pesquisa/perseus-speedrun/scripts")
import em_even7 as E7, em_even as V, em_fit as EF
def ih_cdf(n,x):
    x=Fraction(x)+Fraction(n,2)
    s=Fraction(0);k=0
    while k<=x:
        s+=(-1)**k*math.comb(n,k)*(x-k)**n; k+=1
    return s/math.factorial(n)
HH=33
PI={I:float(ih_cdf(HH,Fraction(2*I+1,2))-ih_cdf(HH,Fraction(2*I-1,2))) for I in range(-16,17)}
print("P(|I|=i):",{i:f"{2*PI[i] if i else PI[0]:.1e}" for i in (0,2,4,6,8,10,12,14,16)})
def grid(K,eps,nd=17):
    d=np.linspace(-eps,eps,nd)
    us=[];ws=[]
    for I in range(-K+1,K):
        us.append(I+d); ws.append(np.full(nd,PI[I]/nd))
    return np.concatenate(us),np.concatenate(ws)
def target(u): return np.sin(2*math.pi*u)/(2*math.pi)
def rms_bits(fn,K,eps):
    u,w=grid(K,eps,65); e=fn(u)-target(u); w=w/w.sum()
    return -math.log2(math.sqrt((w*e*e).sum())), e, u, w
def setmod(K,r,dz):
    E7.K=K;E7.L=float(K);E7.R=r;E7.DZ=dz
    E7.pipeline.__defaults__=(r,);E7.bits.__defaults__=(r,)
def fit_ls(K,dz,r,eps=2.0**-4,floor=1e-10,iters=300):
    setmod(K,r,dz)
    u,w=grid(K,eps); W=np.sqrt(np.maximum(w,floor/len(u)))
    t=target(u); T=E7.basis(u,dz)
    c=E7.fit_weighted(eps,120,dz=dz,r=r)
    lam=1e-3
    f,d=E7.pipeline(c,T,r); cost=((W*(f-t))**2).sum()
    for _ in range(iters):
        J=(W[:,None]*d[:,None])*T
        A=J.T@J; g=J.T@(W*(t-f))
        while True:
            try: dc=np.linalg.solve(A+lam*np.diag(np.diag(A)+1e-30),g)
            except np.linalg.LinAlgError: lam*=10; continue
            f2,d2=E7.pipeline(c+dc,T,r); c2=((W*(f2-t))**2).sum()
            if c2<cost: c=c+dc;f,d=f2,d2;cost=c2;lam=max(lam/3,1e-12);break
            lam*=4
            if lam>1e8: return c
    return c
def show(name,fn,K,c=None):
    b4,e,u,w=rms_bits(fn,K,2.0**-4)
    b2=rms_bits(fn,K,2.0**-2)[0]
    em=[np.abs(e[np.abs(np.round(u))==i]).max() for i in (0,4,6,8,10,12,15) if i<K]
    print(f"{name:26s} rms bits {b4:5.2f} (δ≤2^-4) {b2:5.2f} (δ≤2^-2) | -log2 max: "+" ".join(f"I{i}:{-math.log2(x):4.1f}" for i,x in zip([i for i in (0,4,6,8,10,12,15) if i<K],em)),flush=True)
# referências: recorde e par z7 atual
sh=np.array(EF.SHIPPED)
print("referências (K=16):")
def rec(u):
    return np.array(EF.pipeline(sh,5,u)) if hasattr(EF,"pipeline") else None
import json
ev=json.load(open("/home/zocal/pesquisa/perseus-speedrun/fits/even_k16_z7_r5.json"))
setmod(16,5,7)
show("par z7 r5 atual (minimax)",lambda u:V.pipeline_even(np.array(ev["coeffs"]),5,u,16.0),16)
try:
    print(" funções em_fit:",[x for x in dir(EF) if not x.startswith("_")][:30])
except Exception as e: print(e)
for K,dz,r in ((16,7,5),(16,6,5),(16,5,5),(16,4,5),(16,3,5),(16,7,4),(16,6,4),(16,5,4),(16,4,4)):
    c=fit_ls(K,dz,r)
    lv={3:3,4:3,5:3,6:4,7:4}[dz]
    show(f"LS K{K} z{dz}+{r}DA ({1+lv+r} nív)",lambda u,c=c,r=r,K=K:V.pipeline_even(c,r,u,float(K)),K)
    np.save(f"{sys.argv[1]}/ls_K{K}_z{dz}_r{r}.npy",c)

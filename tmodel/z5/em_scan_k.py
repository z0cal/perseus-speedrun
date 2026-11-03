"""EvalMod par: K menor. Bits (|delta|<=2^-2 e 2^-4) por (K, dz, r) + cauda de |I| (Irwin-Hall, h+1 = 33 parcelas)."""
import math, sys, os, json
import numpy as np
sys.path.insert(0,"/home/zocal/pesquisa/perseus-speedrun/scripts")
import em_even7 as E7
import em_even as V
from fractions import Fraction
# cauda exata de Irwin-Hall n=33 (soma de n uniformes em [-1/2,1/2])
def ih_tail(n,t):
    # P(|S|>t) com S em [-n/2,n/2]; CDF de Irwin-Hall em x=t+n/2, cauda superior por simetria
    x=Fraction(n,2)-Fraction(t).limit_denominator(1000)  # distância ao extremo
    s=Fraction(0)
    k=0
    while k<=x:
        s+=(-1)**k*math.comb(n,k)*(x-k)**n
        k+=1
    f=s/math.factorial(n)
    return 2*float(f)
print("cauda P(|S|>K-1/2), S=soma de 33 uniformes; x N=65536 coef; x 53 refreshes com EvalMod por token")
for K in (16,14,13,12,11,10):
    p=ih_tail(33,K-0.5)
    print(f"  K={K}: P_coef={p:.2e}  por refresh(N)={p*65536:.2e}  por token={p*65536*53:.2e}")
def lvl(dz): return 3 if dz<=5 else (4 if dz<=13 else 5)
def run(K,dz,r):
    E7.K=K; E7.L=float(K); E7.R=r; E7.DZ=dz
    E7.pipeline.__defaults__=(r,)
    E7.bits.__defaults__=(r,)
    c=E7.fit_weighted(E7.EPS_FIT,160,dz=dz,r=r)
    c=E7.refine(c,E7.EPS_FIT,E7.N_FIT,iters=60)
    b2=E7.bits(c,2.0**-2,r); b4=E7.bits(c,2.0**-4,r)
    print(f"K={K:2d} z{dz} + {r} DA: níveis {1+lvl(dz)+r:2d}  bits(2^-2)={b2:5.2f} bits(2^-4)={b4:5.2f}",flush=True)
    return c
for K in (16,14,13,12):
    for dz,r in ((7,5),(6,5),(5,5),(4,5),(7,4),(6,4),(5,4)):
        try: run(K,dz,r)
        except Exception as e: print("erro",K,dz,r,e)

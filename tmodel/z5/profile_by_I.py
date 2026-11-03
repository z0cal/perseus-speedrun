"""Erro do EvalMod por |I| contra a probabilidade exata de cada |I| (soma de 33 uniformes): recorde, z7 e z5.
A densidade de 33 uniformes só vale com o lift centrado em Qd (Lift::Centred). O mod_raise do FIDESlib dá 66 uniformes,
sigma 2,35 (tmodel/z5/kernelhost/census.log), e nele o z5 reprova.
A coluna de amostras por token é um teto: 87 refreshes x 65536 coeficientes, todos densos."""
import math, re
from fractions import Fraction
import numpy as np
from numpy.polynomial import chebyshev as C
Z5=[-0.5745377251677196,-0.916700971342353,0.28591387226312487,-0.027497223315037466,0.00129806776429133,-4.3254981590185784e-05]
Z7=[-5.745215136010838e-01,-9.166784164869499e-01,2.8594548617538085e-01,-2.74681541640594e-02,1.3145350266008343e-03,-3.794693262449959e-05,7.347911536358684e-07,-1.0315035956695626e-08]
src=open("/home/zocal/pesquisa/perseus-speedrun/work/perseus-z5/crates/pc-core/src/bts_setup.rs").read()
SH=[float(x) for x in re.findall(r"-?\d\.\d+(?:e-?\d+)?",re.search(r"EVALMOD_CHEB: \[f64; 15\] = \[(.*?)\];",src,re.S).group(1))]
assert len(SH)==15
def cheb(c,x):
    c=np.array(c,float).copy(); c[0]/=2   # a tabela guarda c0 sem dividir; o avaliador usa c0/2
    return C.chebval(x,c)
def da(x):
    a=(2*math.pi)**(-1/32)
    for _ in range(5):
        x=2*x*x-a*a; a=a*a
    return x
def f_even(c,u):
    w=u/16-1/64
    return da(cheb(c,2*w*w-1))
def f_ship(c,u): return da(cheb(c,u/16))
def t(u): return np.sin(2*math.pi*u)/(2*math.pi)
def cdf(n,x):
    x=Fraction(x)+Fraction(n,2); s=Fraction(0); k=0
    while k<=x:
        s+=(-1)**k*math.comb(n,k)*(x-k)**n; k+=1
    return s/math.factorial(n)
H=33
def dens(n,x):
    x=Fraction(x)+Fraction(n,2); s=Fraction(0); k=0
    while k<=x:
        s+=(-1)**k*math.comb(n,k)*(x-k)**(n-1); k+=1
    return s/math.factorial(n-1)
# I = round(X), X a soma de h = 32 uniformes: P(I = i) é a densidade de 33 uniformes no inteiro i. Conferido em 1,3e9
# coeficientes com os kernels reais do raise e lift centrado (kernelhost/census.log). (cdf(i + 1/2) - cdf(i - 1/2), usado no ajuste, é a densidade de 34.)
P={i:float(dens(H,i)) for i in range(17)}
tail={i:2*sum(P[j] for j in range(i,17)) for i in range(1,17)}
d=np.linspace(-1/16,1/16,65); DRAWS=87*65536
print("|I|  P(|I|=i)   P(|I|>=i)  teto/token(>=i)  recorde  z7   z5  (-log2 do pior erro, |d|<=2^-4)")
for i in range(16):
    us=np.concatenate([i+d,-i+d])
    b=[-math.log2(np.abs(f(c,us)-t(us)).max()) for f,c in ((f_ship,SH),(f_even,Z7),(f_even,Z5))]
    ti=tail[i] if i else 1.0
    print(f"{i:3d}  {P[i]*(2 if i else 1):9.2e}  {ti:9.2e}  {ti*DRAWS:12.2e}     {b[0]:5.2f} {b[1]:5.2f} {b[2]:5.2f}")
for name,f,c in (("recorde",f_ship,SH),("z7",f_even,Z7),("z5",f_even,Z5)):
    parts={}
    for i in range(16):
        us=np.concatenate([i+d,-i+d]); e=f(c,us)-t(us); parts[i]=(e*e).mean()*(P[i]*(2 if i else 1))
    mse=sum(parts.values()); top=sorted(parts,key=parts.get,reverse=True)[:4]
    print(f"{name}: rms {-math.log2(math.sqrt(mse)):.2f} bits; maiores parcelas: "+", ".join(f"|I|={i}:{parts[i]/mse:.1%}" for i in top))
print("P_ABS_I=["+", ".join(f"{P[i]*(2 if i else 1):.6e}" for i in range(16))+"]")

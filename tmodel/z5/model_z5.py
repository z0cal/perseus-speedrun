import json,sys,collections
sys.path.insert(0,"tmodel")
import units_r7 as R, units as U, tokplan as T, r7_estimate as E
C=json.load(open("tmodel/calib.json"))
ms=lambda u:C["tau"]*u[0]+C["kappa"]*u[1]+C["eps"]*u[2]
d=collections.defaultdict(dict)
for l in open(sys.argv[1]):
    j=json.loads(l); d[j["shape"]][j["series"]]=j
tot=collections.Counter()
print("EvalMod por forma (modelo R7, ms):")
for s,v in d.items():
    w=v["recorde"]["weight"]
    a,b,c=[ms(R.evalmod(v[k]["oplist"],True)) for k in ("recorde","par z7","par z5")]
    print(f"  {s:8} peso {w:3.0f}: recorde {a:6.3f}  z7 {b:6.3f} ({b/a:.3f})  z5 {c:6.3f} ({c/a:.3f})")
    tot["rec"]+=w*a; tot["z7"]+=w*b; tot["z5"]+=w*c
print(f"total/token: recorde {tot['rec']:.1f}, z7 {tot['z7']:.1f} ({tot['z7']-tot['rec']:+.1f}), z5 {tot['z5']:.1f} ({tot['z5']-tot['rec']:+.1f})")
# ganho no resto do refresh (raise + CtS) com 1 drop a mais, sites reais do plano
ss=E.sites(sys.argv[2])
T.em_list=lambda top,series,k24:[]
T.NQ=E.NQ
dr=0
for x in ss:
    if x["route"]=="1": continue
    site=dict(x,route="512" if x["route"]=="256" else x["route"])
    a=T.site_units(site,"shipped","shipped")
    s2=dict(site,drop=(site["drop"] or 0)+1)
    b=T.site_units(s2,"shipped","shipped")
    dr+=ms(b)-ms(a)
print(f"raise + CtS com 1 drop a mais em todo refresh com EvalMod: {dr:+.1f} ms/token (modelo do recorde 4)")
for f,n in ((1.0,"bruto"),(968/1086,"x0,89")):
    z7=(tot['z7']-tot['rec'])*f; z5=(tot['z5']-tot['rec'])*f; dd=dr*f
    print(f"{n:6}: z7 {z7:+.1f} ms ({100*z7/968:+.2f} %); z5 só EvalMod {z5:+.1f} ({100*z5/968:+.2f} %); z5 + drop {z5+dd:+.1f} ms ({100*(z5+dd)/968:+.2f} %)")

import collections,json,os,re,sys
TM="/home/zocal/pesquisa/perseus-speedrun/tmodel"; sys.path.insert(0,TM)
import r7_estimate as E, tokplan as T, units as U, units_r7 as R
T.NQ=E.NQ
progs,gdir=sys.argv[1:3]
ss=E.sites(progs); i=0; ref=collections.defaultdict(lambda:[0,0.0])
for b in range(14):
    d=json.load(open(f"{progs}/block_{b}.json"))
    g=json.load(open(f"{gdir}/block_{b}/graph.json"))["nodes"]
    st={n.get("output"):re.sub(r"^blk\d+\.","",n.get("step") or "?") for n in g}
    for s in d["bts_sites"]:
        x=ss[i]; i+=1
        if x["route"]=="1": m=E.SPRU_R7
        else:
            site=dict(x,route="512" if x["route"]=="256" else x["route"])
            T.em_list=lambda top,series,k24:[]
            u=T.site_units(site,"shipped","shipped")
            U.add(u,R.evalmod(E.EMR["Shipped_K16"][E.em_top(x)]["oplist"]),1 if (x["real"] or x["route"]!="stcf") else 2)
            m=E.ms(u)
        k=(st.get(s["var"],"?") if b<13 else "cutmax")+"/"+x["route"]
        ref[k][0]+=1; ref[k][1]+=m
for k,v in sorted(ref.items(),key=lambda a:-a[1][1]): print(f"{k:22s} n={v[0]:3d} {v[1]:7.1f} ms")
print("total",round(sum(v[1] for v in ref.values()),1))

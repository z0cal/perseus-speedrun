"""Por etapa: ms de ops (modelo R7) e refreshes (rota, ms) do recorde 7."""
import collections, json, os, re, sys
TM="/home/zocal/pesquisa/perseus-speedrun/tmodel"
sys.path.insert(0,TM)
import r7_estimate as E
import tokplan as T, units as U, units_r7 as R
T.NQ=E.NQ
progs,gdir=sys.argv[1:3]
DROP=E.DROP
ms=E.ms
ops_by=collections.defaultdict(float); ref_by=collections.defaultdict(lambda:[0,0.0,collections.Counter()])
tot_ops=0
def stepname(g,o,b):
    org=o.get("origin") or {}
    if "Node" in org and org["Node"]<len(g):
        s=g[org["Node"]].get("step") or "?"
        return re.sub(r"^blk\d+\.","",s)
    return "("+next(iter(org),"?")+")"
ss=E.sites(progs)
idx=0
per_block_ref=collections.defaultdict(float); per_block_ops=collections.defaultdict(float)
for b in range(14):
    d=json.load(open(f"{progs}/block_{b}.json"))
    gp=f"{gdir}/block_{b}/graph.json"
    g=json.load(open(gp))["nodes"] if os.path.exists(gp) else []
    V=d["vars"]
    name2=d["names"]
    lv=lambda i: DROP[V[i]["meta"]["level"]]
    last_step="?"
    site_step={}
    def one(o):
        global last_step
        k=T.kind(o)
        if k=="Batched":
            for x in o["kind"]["Batched"]: one(x)
            return
        acc={c:U.z() for c in ("ks","rescale","ew","launch")}
        if k=="MultRescale":
            ins=[lv(i) for i in o["ins"]]; outs=[lv(i) for i in o["outs"]]
            n=E.NQ-max(ins); drop=max(outs)-max(ins)
            for pk in o["pre"]:
                if (next(iter(pk)) if isinstance(pk,dict) else pk)=="Rescale":
                    U.add(acc["rescale"],U.rescale(n+2))
            if len(o["ins"])==2:
                U.add(acc["ks"],R.relin(n,fused=True,drop=max(drop,1)))
            else:
                U.add(acc["ew"],[0,0,3*n]); U.add(acc["rescale"],U.rescale(n))
        else:
            T.op_units(o,lv,acc)
        st=stepname(g,o,b)
        if not st.startswith("("): last_step=st
        val=sum(ms(acc[c]) for c in ("ks","rescale","ew"))
        ops_by[st if b<13 else "cutmax"]+=val
        per_block_ops[b]+=val
    for o in d["ops"]:
        k=T.kind(o)
        if k in("Bootstrap",) :
            pass
        one(o)
    # sites
    for s in d["bts_sites"]:
        x=ss[idx]; idx+=1
        vi=name2.get(s["var"])
        st="?"
        # acha op que produz var
        for o in d["ops"]:
            kk=T.kind(o)
            if kk=="Batched":
                continue
            if vi in o.get("outs",[]):
                st=stepname(g,o,b); break
        if x["route"]=="1":
            m=E.SPRU_R7
        else:
            site=dict(x,route="512" if x["route"]=="256" else x["route"])
            T.em_list=lambda top,series,k24:[]
            u=T.site_units(site,"shipped","shipped")
            chains=1 if (x["real"] or x["route"]!="stcf") else 2
            U.add(u,R.evalmod(E.EMR["Shipped_K16"][E.em_top(x)]["oplist"]),chains)
            m=ms(u)
        key=(st if b<13 else "cutmax")
        ref_by[key][0]+=1; ref_by[key][1]+=m; ref_by[key][2][x["route"]+("r" if x["real"] else "c")]+=1
        per_block_ref[b]+=m
print("OPS total %.1f; REF total %.1f"%(sum(ops_by.values()),sum(v[1] for v in ref_by.values())))
print("\nops por etapa:")
for k,v in sorted(ops_by.items(),key=lambda x:-x[1])[:25]: print(f"  {k:28s} {v:7.1f}")
print("\nrefreshes por etapa (nó da saída do refresh/antecessor):")
for k,v in sorted(ref_by.items(),key=lambda x:-x[1][1]): print(f"  {k:28s} n={v[0]:3d} {v[1]:7.1f} ms {dict(v[2])}")
print("\npor bloco (ops, ref):")
for b in range(14): print(b, round(per_block_ops[b],1), round(per_block_ref[b],1))

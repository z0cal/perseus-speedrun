"""Custo das ops nos planos what-if: níveis finais das variáveis (final_named_levels) + grafo. Só a razão k/k0 importa:
o absoluto é calibrado no plano k0 contra o modelo R7 (353 ms)."""
import collections,json,re,sys
TM="/home/zocal/pesquisa/perseus-speedrun/tmodel"; sys.path.insert(0,TM)
import units as U, units_r7 as R, r7_estimate as E
ms=E.ms
R7="/home/zocal/pesquisa/perseus-speedrun/work/perseus-r7-sweep"
G=f"{R7}/graphs/gpt2_decode_python_n32sp_kc16r_gnabm"
DROP=E.DROP
def limbs(l): return E.NQ-DROP[int(l)]
def cost(name):
    tot=0.0;by=collections.Counter();lv=collections.Counter()
    for b in range(13):
        p=json.load(open(f"{R7}/bootstrap_placements/{name}/block_{b}_placement.json"))
        L=p["summary"]["final_named_levels"]
        nodes=json.load(open(f"{G}/block_{b}/graph.json"))["nodes"]
        for n in nodes:
            t=n["op_type"]; out=n.get("output")
            if out not in L: continue
            ins=[L[i] for i in n.get("inputs",[]) if isinstance(i,str) and i in L]
            lin=max(ins) if ins else L[out]
            x=limbs(min(lin,108))
            st=re.sub(r"^blk\d+\.","",n.get("step") or "?")
            m=0.0
            if t in("mult","square") and (len(ins)==2 or t=="square"):
                m=ms(R.relin(x,fused=True,drop=1 if x%3 else 2))
            elif t in("rotate","conjugate"):
                m=ms(U.ks(x,"n32a4"))
            elif t=="rotate_sum":
                m=ms(U.ks(x,"n32a4"))*1.6
            elif t in("mult",):
                m=ms([0,0,3*x])+ms(U.rescale(x))
            elif t in("add","add_inplace","sub_ct","negate","mult_i"):
                m=ms([0,0,2*x])
            tot+=m;by[st]+=m;lv[st]+=1
    return tot,by
base,bb=cost("slack_r7_k0")
print(f"k0: unidade bruta {base:.1f}; escala para 353 ms: {353/base:.3f}")
f=353/base
for nm in sys.argv[1:]:
    t,by=cost(nm)
    print(f"{nm:30s} ops {t*f:6.1f} ms ({(t-base)*f:+.1f}) | "+" ".join(f"{k}{(by[k]-bb[k])*f:+.1f}" for k in sorted(bb,key=lambda k:-abs(by[k]-bb[k]))[:6]))

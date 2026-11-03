"""Quanto do KL do recorde 7 é aproximação (gêmeo float64) e quanto é ruído (GPU menos gêmeo), nos MESMOS prompts.
Diagnóstico: não ajusta nada nos prompts de dev."""
import glob,json,sys
sys.path.insert(0,"/home/zocal/pesquisa/perseus-speedrun/scripts")
import numpy as np
import kl_eval as K
R7,GPU=sys.argv[1:3]
per={}
for f in sorted(glob.glob(f"{GPU}/out-ab/runs/*/run.json")):
    r=json.load(open(f))
    if r.get("track")!="decode" or r.get("void"): continue
    arm="rec" if "perseus-record" in " ".join(r.get("server_cmd") or []) else "ent"
    for q in (r.get("result") or {}).get("requests",[]):
        if not q.get("warmup"): per.setdefault((arm,q["prompt"]),[]).append(q["kl_scored_mean"])
prompts=json.load(open(f"{R7}/speedrun/prompts/dev.json"))["prompts"]
ids=sorted({k[1] for k in per})
CF=K.pconfig.load_configs(f"{R7}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab/configs.json")
print("prompt  gemeo     GPU_recorde  GPU_entrada")
tw=[];rc=[];en=[]
for i in ids:
    t=np.array(prompts[i][:16])
    e=K.logsoftmax(K.forward_exact(t)); a=K.logsoftmax(K.forward_approx(t,CF))
    k=(np.exp(e)*(e-a)).sum(-1)[8:16].mean()
    r_=np.mean(per.get(("rec",i),[np.nan])); n_=np.mean(per.get(("ent",i),[np.nan]))
    tw.append(k);rc.append(r_);en.append(n_)
    print(f"{i:5d}  {k:.3e}  {r_:.3e}  {n_:.3e}",flush=True)
tw,rc,en=map(np.array,(tw,rc,en))
print(f"médias: gêmeo {np.nanmean(tw):.3e}  recorde {np.nanmean(rc):.3e}  entrada {np.nanmean(en):.3e}")
print(f"ruído aparente (GPU - gêmeo): recorde {np.nanmean(rc-tw):+.3e}  entrada {np.nanmean(en-tw):+.3e}")

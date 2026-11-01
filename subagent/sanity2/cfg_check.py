"""sanity2: Config da sessao (servidor) para o plano da entrada x recorde; fakecap landings; RUNTIME_DROPPED."""
import os, sys, json
from perseus import config as pcfg, _pc
from perseus.serve import gpt2_decode as served
from perseus.examples.gpt2 import env
from perseus.impl.fakecap import parse_landings
E, R = "bootstrap_placements/lnd_fin_e3_em9_cm", "bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm"
ce, cr = served._config(E, "n32"), served._config(R, "n32")
for k in ("bts.evalmod", "model.ln_defer", "model.ln_defer_exit", "chain.preset", "bts.auto_bts_level", "bts.em_fuse_rescale",
          "bts.stc_first", "quirks.fakecap_landings", "model.qkv_lanes", "bts.raise_drops"):
    print(f"{k:28s} entry={ce[k]!r:24s} record={cr[k]!r}")
# every differing field
fe = {f["name"]: ce[f["name"]] for f in _pc.Config.fields()}
fr = {f["name"]: cr[f["name"]] for f in _pc.Config.fields()}
print("fields that differ entry vs record:", {k: (fr[k], fe[k]) for k in fe if fe[k] != fr[k]})
rte, rtr = env.plan_runtime(E, "n32"), env.plan_runtime(R, "n32")
print("env.plan_runtime diff:", {k: (rtr.get(k), rte.get(k)) for k in set(rte) | set(rtr) if rte.get(k) != rtr.get(k)})
# fakecap landings: new Rust path vs the old Python parse, for both configs and for shipped/d27/d27sb
for name, c in (("entry", ce), ("record", cr)):
    old = parse_landings(c["quirks.fakecap_landings"] or "0:32,1:14,256:22,512:22")
    new = dict(_pc.fakecap_landings(c))
    print(f"fakecap landings {name}: old(parse)={old} new(_pc)={new}")
for em in (None, "shipped", "d27", "d27sb"):
    c = served._config(R, "n32"); c["bts.evalmod"] = em
    print("  evalmod", em, "->", dict(_pc.fakecap_landings(c)))
try:
    c = served._config(R, "n32"); c["bts.evalmod"] = "bogus"; print("bogus ->", _pc.fakecap_landings(c))
except Exception as e:
    print("bogus evalmod rejected:", str(e)[:200])

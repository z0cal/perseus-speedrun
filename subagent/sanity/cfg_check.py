"""Sanity: session Config of a plan (as run_decode builds it) -> bts.evalmod, fakecap landings, runtime shadow landings."""
import sys
from perseus import _pc
from perseus import config as pcfg
for plan in sys.argv[1:]:
    cfg = pcfg.plan_runtime(plan, "n32", base=pcfg.preset("n32_gpt2_decode"))
    print(plan, "bts.evalmod =", repr(cfg["bts.evalmod"]), "fakecap_landings =", dict(_pc.fakecap_landings(cfg)))

#!/bin/bash
# Bootstraps of blocks 0-12 on record 4 (work/perseus-r4 = origin/speedrun d97a6f7 build), n32a4 recipe
# (= record recipe + MAX_LEVEL=48 PLAN_REFRESH_ENV_CAP=48, as speedrun/make_n32a4_decode_plan.sh), on the shipped
# merged capture graphs/gpt2_decode_python_n32_kc16rm.
#   plan_count_r4.sh OUT_NAME base|em9 [ilp] [GRAPH]   default placer: min-cut (fast proxy)
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
R4=$W/perseus-r4
cd "$R4"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf CUDA_VISIBLE_DEVICES= PYTHON=$R4/.venv/bin/python
source scripts/local_env.sh
export OMP_NUM_THREADS=1
GR=${4:-graphs/gpt2_decode_python_n32_kc16rm}
RECIPE="$(sed -n 's/^recipe=//p' bootstrap_placements/gpt2_decode_python_n32_kc16r/PLAN_CMD.txt)${N32A4_EXTRA- MAX_LEVEL=48 PLAN_REFRESH_ENV_CAP=48}"
if [ "$2" = em9 ]; then
  # the em9 landings (one CKKS level = 2 primes shallower on every route but SPRU); remap 26:24 only (idempotent
  # under the double application: nothing deliberate is captured at 24 in kc16rm, so the record's 24:14 is a no-op)
  RECIPE=$(echo "$RECIPE" | sed 's/BTS_LEVEL=34 /BTS_LEVEL=32 /; s/BTS_LEVEL=28 /BTS_LEVEL=26 /; s/PLAN_SPARSE_BTS_OUT=1:16,512:26/PLAN_SPARSE_BTS_OUT=1:16,512:24/; s/PLAN_DELIB_LANDING_REMAP=24:14/PLAN_DELIB_LANDING_REMAP=26:24/; s/PLAN_STC_FIRST_FALLBACK=34:1/PLAN_STC_FIRST_FALLBACK=32:1/')
fi
[ "${3:-}" = ilp ] || RECIPE=$(echo "$RECIPE" | sed 's/PLAN_PLACER=[^ ]*//; s/PLAN_ILP_TIME_LIMIT=[^ ]*//; s/PLAN_ILP_TIEBREAK=[^ ]*//')
echo "$RECIPE" > /home/zocal/pesquisa/perseus-speedrun/subagent/r4/recipe_$1.txt
# shellcheck disable=SC2086
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$1" blocks > /home/zocal/pesquisa/perseus-speedrun/subagent/r4/log_$1.txt 2>&1
"$PYTHON" - "bootstrap_placements/$1" <<'PY'
import glob, json, os, sys
per = {int(os.path.basename(f)[6:-15]): json.load(open(f))["summary"] for f in glob.glob(f"{sys.argv[1]}/block_*_placement.json") if "feedback" not in f}
t = [per[b]["total_bootstraps"] for b in sorted(per)]
print(sys.argv[1], "per stage", t, "blocks0-12", sum(t[:13]), "b11 exit", per[11]["exit_level"], per[11].get("exit_deg"), "b12 exit", per[12]["exit_level"], per[12].get("exit_deg"))
PY

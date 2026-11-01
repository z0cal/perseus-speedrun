#!/bin/bash
# Bootstraps of blocks 0-12 for a capture under the em9 recipe (perseus = the evalmod-d27 build).
#   plan_count.sh CAPTURE_DIR OUT_NAME [ilp]     default placer: min-cut (fast proxy); "ilp": the recipe's exact ILP
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
cd "$W/perseus"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf CUDA_VISIBLE_DEVICES= PYTHON=$W/perseus/.venv/bin/python
source scripts/local_env.sh
RECIPE=$(sed -n 's/^recipe=//p' "$W/../plans/gpt2_decode_em9/PLAN_CMD.txt")
[ "${3:-}" = ilp ] || RECIPE=$(echo "$RECIPE" | sed 's/PLAN_PLACER=[^ ]*//; s/PLAN_ILP_TIME_LIMIT=[^ ]*//; s/PLAN_ILP_TIEBREAK=[^ ]*//')
# shellcheck disable=SC2086
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh "$1" "$2" > "$W/logs_$2.txt" 2>&1
"$PYTHON" - "bootstrap_placements/$2" <<'PY'
import glob, json, os, sys
per = {os.path.basename(f)[:-15]: json.load(open(f))["summary"]["total_bootstraps"]
       for f in glob.glob(f"{sys.argv[1]}/block_*_placement.json")}
print(sys.argv[1], "blocks0-12", sum(v for k, v in per.items() if k not in ("block_13", "block_0_feedback")))
PY

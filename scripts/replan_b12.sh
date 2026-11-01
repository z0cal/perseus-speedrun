#!/bin/bash
# Replan only block 12 of a plan with the tail's refresh envelope capped lower (so CutMax's stage 13 has room above its
# entry), as speedrun/make_tail_plan.sh plans a single stage; then plan stage 13 by speedrun/make_cutmax_plan.sh.
#   bash replan_b12.sh SRC_PLAN NEW_PLAN [TAIL_CAP]      e.g. fin_dfr_m fin_dfr_m2 46
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
SRC=$1 NEW=$2 CAPV=${3:-46}
cd "$W/perseus"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf CUDA_VISIBLE_DEVICES= PYTHON=$W/perseus/.venv/bin/python
source scripts/local_env.sh
GR=$(sed -n 's/^graph=//p' "bootstrap_placements/$SRC/PLAN_CMD.txt")
CFG=$(sed -n 's/^configs=//p' "bootstrap_placements/$SRC/PLAN_CMD.txt")
RECIPE="$(sed -n 's/^recipe=//p' "bootstrap_placements/$SRC/PLAN_CMD.txt" | sed "s/tail:48/tail:$CAPV/g; s/PLAN_REFRESH_ENV_CAP=50/PLAN_REFRESH_ENV_CAP=$CAPV/g") ${B12_EXTRA:-}"
test ! -e "bootstrap_placements/$NEW" && cp -r "bootstrap_placements/$SRC" "bootstrap_placements/$NEW"
sed -i "s/tail:48/tail:$CAPV/g; s/PLAN_REFRESH_ENV_CAP=50 /PLAN_REFRESH_ENV_CAP=50 PLAN_B12_ENV_CAP=$CAPV /" "bootstrap_placements/$NEW/PLAN_CMD.txt"
read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$SRC/block_11_placement.json")
TMP="$(mktemp -d graphs/_b12_XXXX)"
ln -s "$PWD/$GR/block_12" "$TMP/block_12" && cp "$GR/capture_env.json" "$TMP/"
# shellcheck disable=SC2086
env $RECIPE GRAPH_DIR="$TMP" OUT_NAME="${NEW}_b12" FIRST_ENTRY_LEVEL="$L" FIRST_ENTRY_DEG="$D" \
    systemd-run --user --scope -q -p MemoryMax=7G bash python/perseus/plan/run_bootstrap_all_blocks.sh > "$W/${NEW}_b12.log" 2>&1
cp "bootstrap_placements/${NEW}_b12/block_12_placement.json" "bootstrap_placements/$NEW/"
"$PYTHON" - "$GR" "bootstrap_placements/$NEW/block_12_placement.json" <<'PY'
import json, os, sys
from perseus.plan import contract
contract.stamp_file(sys.argv[2], json.load(open(os.path.join(sys.argv[1], "capture_env.json"))))
s = json.load(open(sys.argv[2]))["summary"]
print("block 12:", s["total_bootstraps"], "bootstraps, exit", s["exit_level"], s.get("exit_deg"))
PY
CUTMAX_GRAPH=graphs/gpt2_decode_python_n32_kc16_cm systemd-run --user --scope -q -p MemoryMax=7G \
    bash speedrun/make_cutmax_plan.sh "$NEW" "${NEW}_cm" "$CFG" > "$W/${NEW}_cm.log" 2>&1 || { tail -3 "$W/${NEW}_cm.log"; exit 1; }
"$PYTHON" - "bootstrap_placements/${NEW}_cm" <<'PY'
import glob, json, os, sys
per = {os.path.basename(f)[:-15]: json.load(open(f))["summary"]["total_bootstraps"]
       for f in glob.glob(f"{sys.argv[1]}/block_*_placement.json")}
b = sum(v for k, v in per.items() if k not in ("block_13", "block_0_feedback"))
print(sys.argv[1], "blocks0-12", b, "block13", per.get("block_13"), "per token", b + per.get("block_13", 0))
PY

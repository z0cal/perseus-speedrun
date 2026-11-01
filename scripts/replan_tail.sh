#!/bin/bash
# Replan blocks 11 and 12 of a plan one at a time, each as its own (final) block so the ILP's cap_exit keeps its exit
# no deeper than the min-cut's, then plan CutMax's stage 13 (speedrun/make_cutmax_plan.sh). The full-chain ILP lets
# block 11 exit deep (it books the saving, the unplanned argmax pays), which left CutMax no room above its entry.
#   bash replan_tail.sh SRC_PLAN NEW_PLAN
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
SRC=$1 NEW=$2
cd "$W/perseus"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf CUDA_VISIBLE_DEVICES= PYTHON=$W/perseus/.venv/bin/python
source scripts/local_env.sh
GR=$(sed -n 's/^graph=//p' "bootstrap_placements/$SRC/PLAN_CMD.txt")
CFG=$(sed -n 's/^configs=//p' "bootstrap_placements/$SRC/PLAN_CMD.txt")
RECIPE="$(sed -n 's/^recipe=//p' "bootstrap_placements/$SRC/PLAN_CMD.txt")"
test ! -e "bootstrap_placements/$NEW" && cp -r "bootstrap_placements/$SRC" "bootstrap_placements/$NEW"
for b in 11 12; do
    read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$NEW/block_$((b - 1))_placement.json")
    TMP="$(mktemp -d graphs/_b${b}_XXXX)"
    ln -s "$PWD/$GR/block_$b" "$TMP/block_$b" && cp "$GR/capture_env.json" "$TMP/"
    # shellcheck disable=SC2086
    env $RECIPE GRAPH_DIR="$TMP" OUT_NAME="${NEW}_b$b" FIRST_ENTRY_LEVEL="$L" FIRST_ENTRY_DEG="$D" \
        systemd-run --user --scope -q -p MemoryMax=7G bash python/perseus/plan/run_bootstrap_all_blocks.sh \
        > "$W/${NEW}_b$b.log" 2>&1
    cp "bootstrap_placements/${NEW}_b$b/block_${b}_placement.json" "bootstrap_placements/$NEW/"
    "$PYTHON" - "$GR" "bootstrap_placements/$NEW/block_${b}_placement.json" "$b" <<'PY'
import json, os, sys
from perseus.plan import contract
contract.stamp_file(sys.argv[2], json.load(open(os.path.join(sys.argv[1], "capture_env.json"))))
s = json.load(open(sys.argv[2]))["summary"]
print(f"block {sys.argv[3]}: {s['total_bootstraps']} bootstraps, exit {s['exit_level']}:{s.get('exit_deg')}")
PY
done
CUTMAX_GRAPH=graphs/gpt2_decode_python_n32_kc16_cm systemd-run --user --scope -q -p MemoryMax=7G \
    bash speedrun/make_cutmax_plan.sh "$NEW" "${NEW}_cm" "$CFG" > "$W/${NEW}_cm.log" 2>&1 || { tail -3 "$W/${NEW}_cm.log"; exit 1; }
"$PYTHON" - "bootstrap_placements/${NEW}_cm" <<'PY'
import glob, json, os, sys
per = {os.path.basename(f)[:-15]: json.load(open(f))["summary"]["total_bootstraps"]
       for f in glob.glob(f"{sys.argv[1]}/block_*_placement.json")}
b = sum(v for k, v in per.items() if k not in ("block_13", "block_0_feedback"))
print(sys.argv[1], "blocks0-12", b, "block13", per.get("block_13"), "per token", b + per.get("block_13", 0))
PY

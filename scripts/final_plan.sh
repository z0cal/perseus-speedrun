#!/bin/bash
# The candidate's full Decode-8 plan in the em9 build (work/perseus: evalmod-d27 + defer-norm), as the record's own
# recipe does it: position-0 capture -> base plan (ILP, blocks + argmax + feedback) -> merged-magnitude capture over
# 4 pool rows x 16 positions (speedrun/make_decode_capture.sh) -> final plan on it. One job at a time, memory-capped.
#   bash final_plan.sh [NAME] [CONFIGS]      (defaults: dfr, configs/model/approximation/cand_dfr_cm/configs.json)
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
NAME=${1:-dfr}
CFG=${2:-configs/model/approximation/cand_dfr_cm/configs.json}
cd "$W/perseus"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf CUDA_VISIBLE_DEVICES= PYTHON=$W/perseus/.venv/bin/python
source scripts/local_env.sh
CAP=graphs/fin_${NAME} CAPM=graphs/fin_${NAME}_m
EM9=$W/../plans/gpt2_decode_em9_cm/PLAN_CMD.txt
RUNTIME=$(sed -n 's/^runtime=//p' "$EM9")
RECIPE=$(sed -n 's/^recipe=//p' "$EM9")
mk_plan_dir() {   # plan dir $1 on graph $2
    mkdir -p "bootstrap_placements/$1"
    { echo "graph=$2"
      echo "tool=python/perseus/examples/gpt2/make_plan.sh (blocks, argmax, then feedback)"
      echo "recipe=$RECIPE"
      echo "runtime=$RUNTIME"
      echo "configs=$CFG"; } > "bootstrap_placements/$1/PLAN_CMD.txt"
}
cap() { systemd-run --user --scope -q -p MemoryMax=7G "$@"; }

echo "[1/4] position-0 capture $(date +%H:%M)"
if [ ! -d "$CAP/block_13" ]; then
    RT=$(sed -n 's/^runtime=//p' bootstrap_placements/gpt2_decode_python_n32_m16i/PLAN_CMD.txt | tr ' ' '\n' | grep = | sed 's/^/--set /')
    # shellcheck disable=SC2086
    cap "$PYTHON" -m perseus.examples.gpt2.run_decode --fake --tokens 1 --argmax --capture "$CAP" --configs "$CFG" \
        $RT --set SM_CHEB_GS=1 > "$W/fin_${NAME}_cap.log" 2>&1
fi
tail -1 "$W/fin_${NAME}_cap.log"

echo "[2/4] base plan (ILP) $(date +%H:%M)"
mk_plan_dir "fin_${NAME}" "$CAP"
[ -f "bootstrap_placements/fin_${NAME}/block_12_placement.json" ] || cap bash scripts/make_plans.sh "fin_${NAME}"

echo "[3/4] merged-magnitude capture $(date +%H:%M)"
if [ ! -d "$CAPM/block_13" ]; then
    BASE="$CAP" PLAN="bootstrap_placements/fin_${NAME}" JOBS=2 POOL=$HF_HOME/perseus/pools/openwebtext_gpt2_2000000.npy \
        systemd-run --user --scope -q -p MemoryMax=11G bash speedrun/make_decode_capture.sh "$CAPM" > "$W/fin_${NAME}_merge.log" 2>&1
fi
tail -2 "$W/fin_${NAME}_merge.log"

echo "[4/4] final plan (ILP blocks 0-12, then CutMax stage 13 as the record does) $(date +%H:%M)"
mk_plan_dir "fin_${NAME}_m" "$CAPM"
# blocks only: the position-0 / merged captures carry no logit-row magnitudes for CutMax (its stage is planned by
# speedrun/make_cutmax_plan.sh from the record's stage-13 capture, as for the record and gpt2_decode_em9_cm)
if [ ! -f "bootstrap_placements/fin_${NAME}_m/block_12_placement.json" ]; then
    # shellcheck disable=SC2086
    env $RECIPE systemd-run --user --scope -q -p MemoryMax=7G bash python/perseus/examples/gpt2/make_plan.sh "$CAPM" "fin_${NAME}_m" > "$W/fin_${NAME}_m_blocks.log" 2>&1
fi
CUTMAX_GRAPH=graphs/gpt2_decode_python_n32_kc16_cm cap bash speedrun/make_cutmax_plan.sh "fin_${NAME}_m" "fin_${NAME}_m_cm" \
    "$CFG" > "$W/fin_${NAME}_m_cm.log" 2>&1
tail -2 "$W/fin_${NAME}_m_cm.log"
"$PYTHON" - "bootstrap_placements/fin_${NAME}" "bootstrap_placements/fin_${NAME}_m_cm" <<'PY'
import glob, json, os, sys
for d in sys.argv[1:]:
    per = {os.path.basename(f)[:-15]: json.load(open(f))["summary"]["total_bootstraps"]
           for f in glob.glob(f"{d}/block_*_placement.json")}
    print(d, "total", sum(per.values()), "blocks0-12",
          sum(v for k, v in per.items() if k not in ("block_13", "block_0_feedback")), "block13", per.get("block_13"),
          "per token", sum(v for k, v in per.items() if k != "block_0_feedback"))
PY
echo "done $(date +%H:%M)"

#!/bin/bash
# speedrun/make_n32a4_decode_plan.sh (record 4) as a copy that runs in work/perseus-r4 with:
#   MODE=base|em9|d27sb   em9 = the d27 EvalMod's landings in the recipe (BTS_LEVEL 34->32, 28->26, 512:26->24,
#                   STC_FIRST_FALLBACK 34:1->32:1, remap 24:14 -> 26:24 [idempotent: nothing deliberate at 24]) and
#                   FIDESLIB_BTS_EVALMOD=d27 on the runtime line
#   PLACER=mincut|ilp  for stages 0-12 (CutMax's stage 13 always on the ILP, as the record)
# and no rm: its temporaries are moved aside (_b12 dirs kept).
#   bash plan_n32a4.sh OUT_NAME      (OUT_NAME_cm is the full plan)
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
R4=$W/perseus-r4
cd "$R4"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf PYTHON=$R4/.venv/bin/python
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
OUT=${1:?out}
MODE=${MODE:-em9} PLACER=${PLACER:-mincut}
BASE=gpt2_decode_python_n32_kc16r GR=${GR:-graphs/gpt2_decode_python_n32_kc16rm}
CUTMAX_GRAPH=${CUTMAX_GRAPH:-graphs/gpt2_decode_python_n32_kc16r_cm}
CONFIGS=${CONFIGS:-configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json}
CAP=${CAP:-46}            # the ILP exit cap passed to the planner
CHECK=${CHECK:-46}        # the absolute effective exit CutMax needs (the record script checks CAP itself)
B12=${OUT}_b12${B12SUF:-}
RECIPE="$(sed -n 's/^recipe=//p' "bootstrap_placements/$BASE/PLAN_CMD.txt") MAX_LEVEL=48 PLAN_REFRESH_ENV_CAP=48"
EXTRA_RT="CHAIN_PRESET=n32a4"
if [ "$MODE" = em9 ]; then
  RECIPE=$(echo "$RECIPE" | sed 's/BTS_LEVEL=34 /BTS_LEVEL=32 /; s/BTS_LEVEL=28 /BTS_LEVEL=26 /; s/PLAN_SPARSE_BTS_OUT=1:16,512:26/PLAN_SPARSE_BTS_OUT=1:16,512:24/; s/PLAN_DELIB_LANDING_REMAP=24:14/PLAN_DELIB_LANDING_REMAP=26:24/; s/PLAN_STC_FIRST_FALLBACK=34:1/PLAN_STC_FIRST_FALLBACK=32:1/')
  EXTRA_RT="CHAIN_PRESET=n32a4 FIDESLIB_BTS_EVALMOD=d27"
elif [ "$MODE" = d27sb ]; then
  # d27 on the sparse routes only: SparseB lands one level shallower (512:26 -> 24), dense / StC-first and the
  # deliberate remap as the record
  RECIPE=$(echo "$RECIPE" | sed 's/PLAN_SPARSE_BTS_OUT=1:16,512:26/PLAN_SPARSE_BTS_OUT=1:16,512:24/')
  EXTRA_RT="CHAIN_PRESET=n32a4 FIDESLIB_BTS_EVALMOD=d27sb"
fi
# RT_ADD: model switches the capture ran under, onto the runtime= line (the served decode re-records its graphs under
# that line: e.g. RT_ADD=LN_DEFER=1 for an LN-deferred capture, or it records the shipped LayerNorm)
EXTRA_RT="$EXTRA_RT${RT_ADD:+ $RT_ADD}"
BRECIPE=$RECIPE
[ "$PLACER" = ilp ] || BRECIPE=$(echo "$RECIPE" | sed 's/PLAN_PLACER=[^ ]*//; s/PLAN_ILP_TIME_LIMIT=[^ ]*//; s/PLAN_ILP_TIEBREAK=[^ ]*//')
echo "[$(date +%H:%M:%S)] blocks ($MODE, $PLACER)"
if [ "${RESUME:-0}" != 1 ]; then
# shellcheck disable=SC2086
env $BRECIPE bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$OUT" blocks
fi
if [ -n "${B12_FROM:-}" ]; then
  # stage 12 taken from another stage-12-alone pass planned at the same entry (checked), not re-planned
  read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$OUT/block_11_placement.json")
  [ "$L $D" = "${B12_ENTRY:?entry the B12_FROM pass was planned at}" ] || { echo "block 11 exits at $L $D, not $B12_ENTRY"; exit 2; }
  mkdir -p "bootstrap_placements/$B12"
  # resume-safe: the blocks pass's own stage 12 and argmax stage 13 are set aside once, never overwritten
  [ -e "bootstrap_placements/$B12/block_12_placement.blocks_pass.json" ] || mv "bootstrap_placements/$OUT/block_12_placement.json" "bootstrap_placements/$B12/block_12_placement.blocks_pass.json"
  cp "$B12_FROM" "bootstrap_placements/$OUT/block_12_placement.json"
  [ ! -e "bootstrap_placements/$OUT/block_13_placement.json" ] || mv "bootstrap_placements/$OUT/block_13_placement.json" "bootstrap_placements/$B12/block_13_placement.argmax_dropped.json"
  echo "stage 12 from $B12_FROM (entry $L/d$D)"
else
read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$OUT/block_11_placement.json")
TMP="$(mktemp -d graphs/_b12_XXXX)"
ln -s "$PWD/$GR/block_12" "$TMP/block_12" && cp "$GR/capture_env.json" "$TMP/"
echo "[$(date +%H:%M:%S)] stage 12 alone at $L/d$D, exit cap $CAP"
# shellcheck disable=SC2086
env $BRECIPE BLOCKS_ENTRY_LEVEL="$L" BLOCKS_ENTRY_DEG="$D" PLAN_ILP_EXIT_CAP="$CAP" \
    bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$B12" blocks
"$PYTHON" - "bootstrap_placements/$B12/block_12_placement.json" "$CHECK" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
eff = s["exit_level"] + (2 if (s.get("exit_deg") or 1) == 2 else 0)
print("stage 12 alone:", s["total_bootstraps"], "exit", s["exit_level"], s.get("exit_deg"))
if eff > float(sys.argv[2]):
    sys.exit(f"stage 12 exits at {s['exit_level']}/d{s.get('exit_deg')}, past the cap {sys.argv[2]}")
PY
# resume-safe: the blocks pass's own stage 12 and argmax stage 13 are set aside once, never overwritten
[ -e "bootstrap_placements/$B12/block_12_placement.blocks_pass.json" ] || mv "bootstrap_placements/$OUT/block_12_placement.json" "bootstrap_placements/$B12/block_12_placement.blocks_pass.json"
cp "bootstrap_placements/$B12/block_12_placement.json" "bootstrap_placements/$OUT/"
[ ! -e "bootstrap_placements/$OUT/block_13_placement.json" ] || mv "bootstrap_placements/$OUT/block_13_placement.json" "bootstrap_placements/$B12/block_13_placement.argmax_dropped.json"
fi
echo "[$(date +%H:%M:%S)] feedback"
# shellcheck disable=SC2086
env $BRECIPE bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$OUT" feedback
TOOL="subagent/r4/plan_n32a4.sh (MODE=$MODE PLACER=$PLACER; copy of speedrun/make_n32a4_decode_plan.sh)"
# CONFIGS other than the record's: the plan binds those (make_cutmax_plan.sh checks the configs= line against them)
CFG_LINE="configs=$CONFIGS"
[ "$CONFIGS" = configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json ] && CFG_LINE=
sed -e "s|^tool=.*|tool=$TOOL|" -e "s|^recipe=.*|recipe=$RECIPE|" -e "s|^runtime=\(.*\)|runtime=\1 $EXTRA_RT|" \
    -e "s|^graph=[^ ]*|graph=$GR|" ${CFG_LINE:+-e "s|^configs=.*|$CFG_LINE|"} "bootstrap_placements/$BASE/PLAN_CMD.txt" > "bootstrap_placements/$OUT/PLAN_CMD.txt"
grep "^capture13=" "bootstrap_placements/${BASE}_cm/PLAN_CMD.txt" >> "bootstrap_placements/$OUT/PLAN_CMD.txt" || true
echo "[$(date +%H:%M:%S)] CutMax stage 13 (ILP)"
CUTMAX_PLACER=ilp CUTMAX_GRAPH="$CUTMAX_GRAPH" bash speedrun/make_cutmax_plan.sh "$OUT" "${OUT}_cm" "$CONFIGS"
"$PYTHON" - bootstrap_placements "${OUT}_cm" <<'PY'
import json, sys
root, name = sys.argv[1:]
tot = [json.load(open(f"{root}/{name}/block_{b}_placement.json"))["summary"]["total_bootstraps"] for b in range(14)]
print(f"{name}: refreshes {tot}, stages 0-12 {sum(tot[:13])}, all {sum(tot)}")
PY
echo "[$(date +%H:%M:%S)] done"

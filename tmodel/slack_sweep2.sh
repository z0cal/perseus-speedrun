#!/bin/bash
# How many levels per block does the ILP need to drop a refresh? Record 4's recipe and merged capture, every refresh
# landing k CKKS levels (2k limbs) shallower than the runtime can (dense/StC-first, SparseB, SPRU and the deliberate K/V
# push): a planner-only what-if (the plans do not compile). Blocks 0-12 pass only, ILP, one at a time. CPU only.
#   KS="1 2 3" bash tmodel/slack_sweep2.sh      -> bootstrap_placements/slack_k<k>/, log tmodel/slack_sweep.log
#   ROUTES="sb" (any of dense sb spru delib; default all): only those routes' landings shift -> slack_k<k>_<routes>
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
R4=$W/perseus-r4
LOG=~/pesquisa/perseus-speedrun/tmodel/slack_sweep2.log
cd "$R4"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf PYTHON=$R4/.venv/bin/python
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
GR=${GR:-graphs/gpt2_decode_python_n32_kc16rm}
BASE_RECIPE="$(sed -n 's/^recipe=//p' bootstrap_placements/gpt2_decode_python_n32_kc16r/PLAN_CMD.txt) MAX_LEVEL=48 PLAN_REFRESH_ENV_CAP=48"
for k in ${KS:-1 2 3}; do
  s=$((2 * k)); RT=${ROUTES:-dense sb spru delib}
  sh() { case " $RT " in *" $1 "*) echo "$s";; *) echo 0;; esac; }
  d=$(sh dense) b=$(sh sb) p=$(sh spru) l=$(sh delib)
  R=$(echo "$BASE_RECIPE" | sed "s/BTS_LEVEL=28 /BTS_LEVEL=$((28 - d)) /; s/BTS_LEVEL=34 /BTS_LEVEL=$((34 - d)) /; \
      s/PLAN_SPARSE_BTS_OUT=1:16,512:26/PLAN_SPARSE_BTS_OUT=1:$((16 - p)),512:$((26 - b))/; \
      s/PLAN_DELIB_LANDING_REMAP=24:14/PLAN_DELIB_LANDING_REMAP=26:$((26 - l))/; \
      s/PLAN_STC_FIRST_FALLBACK=34:1/PLAN_STC_FIRST_FALLBACK=$((34 - d)):1/")
  N=slack_k$k${ROUTES:+_$(echo "$ROUTES" | tr ' ' '_')}
  echo "[$(date +%H:%M:%S)] $N: $(echo "$R" | tr ' ' '\n' | grep -E '^(BTS_LEVEL|PLAN_SPARSE_BTS_OUT|PLAN_DELIB_LANDING_REMAP|PLAN_STC_FIRST_FALLBACK)=' | tr '\n' ' ')" >> "$LOG"
  # shellcheck disable=SC2086
  systemd-run --user --scope -p MemoryMax=6G --quiet env $R bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$N" blocks \
      > "$LOG.$N" 2>&1 || echo "$N: make_plan failed (see $LOG.$N)" >> "$LOG"
  "$PYTHON" - "bootstrap_placements/$N" >> "$LOG" <<'PY'
import collections, json, sys
d = sys.argv[1]
tot, mix = [], collections.Counter()
for b in range(13):
    p = json.load(open(f"{d}/block_{b}_placement.json"))
    tot.append(p["summary"]["total_bootstraps"])
    mix.update(p.get("sparse_slots", {}).values())
print(f"  {d}: blocks 0-12 {sum(tot)} {tot}, sparse sites {dict(mix)}")
PY
done
echo "[$(date +%H:%M:%S)] done" >> "$LOG"

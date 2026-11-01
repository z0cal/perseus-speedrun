#!/bin/bash
# LN-deferred (LN_DEFER=1) captures of approximation config CFG and the ILP's blocks 0-12 pass on the record's recipe
# (shipped EvalMod, no slack): how many refreshes. CPU only, one heavy job at a time.
#   CFG=r4_ln3_smrel_db bash tmodel/run_lndcfg.sh      (log tmodel/run_lnd_$CFG.log; graphs lnd_$CFG{,m})
set -euo pipefail
S=~/pesquisa/perseus-speedrun
R4=$S/work/perseus-r4
N=${CFG:?config name}
G=lnd_$N
LOG=$S/tmodel/run_$G.log
cd "$R4"
export PERSEUS_DATA=$S/work/data HF_HOME=$S/work/hf PYTHON=$R4/.venv/bin/python LN_DEFER=1
echo "[$(date +%H:%M:%S)] position-0 capture" >> "$LOG"
CFG=python/perseus/configs/model/approximation/$N/configs.json bash ../lndefer/cap.sh graphs/$G >> "$LOG" 2>&1
mkdir -p bootstrap_placements/capplan_$G
{ echo "$(grep '^runtime=' bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm/PLAN_CMD.txt) LN_DEFER=1"
  echo "configs=configs/model/approximation/$N/configs.json"; } > bootstrap_placements/capplan_$G/PLAN_CMD.txt
echo "[$(date +%H:%M:%S)] merged capture" >> "$LOG"
systemd-run --user --scope -p MemoryMax=7G --quiet env LN_DEFER=1 BASE=graphs/$G PLAN=bootstrap_placements/capplan_$G JOBS=1 \
    bash speedrun/make_decode_capture.sh graphs/${G}m >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] blocks pass" >> "$LOG"
cd "$S"
GR=graphs/${G}m ROUTES=$G KS=0 bash tmodel/slack_sweep2.sh
grep -A1 "slack_k0_$G:" tmodel/slack_sweep2.log | tail -1 >> "$LOG"
echo "[$(date +%H:%M:%S)] done" >> "$LOG"

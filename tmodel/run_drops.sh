#!/bin/bash
# Record 4's graph, ILP in ms (recalibrated route costs) with wider raise drops: SparseB / StC-first drops up to 7 and
# landings down to 44 (record: drops 1-5, landing max 42). CPU only. Waits for WAIT_LOG to hold an EXIT= line.
#   WAIT_LOG=tmodel/run_r4_ln3_smrel_gelubs.log bash tmodel/run_drops.sh     (log: tmodel/run_drops.log)
set -euo pipefail
S=~/pesquisa/perseus-speedrun
LOG=$S/tmodel/run_drops.log
until grep -q 'EXIT=' "$S/${WAIT_LOG:?log to wait for}" 2>/dev/null; do sleep 30; done
cd "$S"
echo "[$(date +%H:%M:%S)] plan msdrop7" >> "$LOG"
systemd-run --user --scope -p MemoryMax=6G --quiet env PLAN_ILP_OBJECTIVE=ms PLAN_BTS_MS=0:11.9,512:8.2,1:8.0 MODE=base \
    PLACER=ilp RECIPE_ADD="PLAN_RAISE_DROP_SET=1,2,3,4,5,6,7 PLAN_RAISE_LANDING_MAX=44" \
    bash tmodel/plan_n32a4_add.sh msdrop7 >> "$LOG" 2>&1
P=$S/work/perseus-r4/bootstrap_placements/msdrop7_cm
sed -i 's/FIDESLIB_BTS_RAISE_DROPS=1,2,3,4,5 /FIDESLIB_BTS_RAISE_DROPS=1,2,3,4,5,6,7 /' "$P/PLAN_CMD.txt"
echo "recipe_env=PLAN_ILP_OBJECTIVE=ms PLAN_BTS_MS=0:11.9,512:8.2,1:8.0 RECIPE_ADD=\"PLAN_RAISE_DROP_SET=1,2,3,4,5,6,7 PLAN_RAISE_LANDING_MAX=44\" (tmodel/run_drops.sh; runtime drops 1-7)" >> "$P/PLAN_CMD.txt"
echo "[$(date +%H:%M:%S)] done" >> "$LOG"

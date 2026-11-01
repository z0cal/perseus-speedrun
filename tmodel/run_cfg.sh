#!/bin/bash
# Record 4 with approximation config CFG (python/perseus/configs/model/approximation/CFG): position-0 + merged capture,
# full plan (record recipe, ILP), compile_check + route_mix. CPU only. Waits for WAIT_LOG to hold a RUN_*EXIT= line.
#   CFG=r4_ln3_smrel_gelubs WAIT_LOG=tmodel/run_ln3.log bash tmodel/run_cfg.sh   (log: tmodel/run_$CFG.log)
set -euo pipefail
S=~/pesquisa/perseus-speedrun
R4=$S/work/perseus-r4
N=${CFG:?config name}
LOG=$S/tmodel/run_$N.log
until grep -q 'EXIT=' "$S/${WAIT_LOG:?log to wait for}" 2>/dev/null; do sleep 30; done
cd "$R4"
export PERSEUS_DATA=$S/work/data HF_HOME=$S/work/hf PYTHON=$R4/.venv/bin/python
echo "[$(date +%H:%M:%S)] position-0 capture" >> "$LOG"
CFG=python/perseus/configs/model/approximation/$N/configs.json bash ../lndefer/cap.sh graphs/$N >> "$LOG" 2>&1
mkdir -p bootstrap_placements/capplan_$N
{ grep "^runtime=" bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm/PLAN_CMD.txt; echo "configs=configs/model/approximation/$N/configs.json"; } > bootstrap_placements/capplan_$N/PLAN_CMD.txt
echo "[$(date +%H:%M:%S)] merged capture" >> "$LOG"
systemd-run --user --scope -p MemoryMax=7G --quiet env BASE=graphs/$N PLAN=bootstrap_placements/capplan_$N JOBS=1 \
    bash speedrun/make_decode_capture.sh graphs/${N}m >> "$LOG" 2>&1
C=$S/tmodel/graphs_combo/$N
mkdir -p "$C"
for b in $(seq 0 12); do ln -sfn "$R4/graphs/${N}m/block_$b" "$C/block_$b"; done
ln -sfn "$R4/graphs/gpt2_decode_python_n32_kc16r_cm/block_13" "$C/block_13"
cp graphs/${N}m/capture_config.json graphs/${N}m/capture_env.json "$C/"
echo "[$(date +%H:%M:%S)] plan" >> "$LOG"
cd "$S"
systemd-run --user --scope -p MemoryMax=6G --quiet env MODE=base PLACER=ilp GR=graphs/${N}m \
    CONFIGS=configs/model/approximation/$N/configs.json bash subagent/r4/plan_n32a4.sh $N >> "$LOG" 2>&1
cd "$R4"
export PYTHONPATH=python CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
echo "[$(date +%H:%M:%S)] compile_check" >> "$LOG"
"$PYTHON" "$S/subagent/audit/compile_check.py" "$C" bootstrap_placements/${N}_cm $(printf 'block_%d ' $(seq 0 13)) 2>&1 | tail -16 >> "$LOG"
echo "[$(date +%H:%M:%S)] route_mix" >> "$LOG"
"$PYTHON" "$S/subagent/r4/route_mix_r4.py" "$C" bootstrap_placements/${N}_cm >> "$LOG" 2>&1 || true
echo "[$(date +%H:%M:%S)] done" >> "$LOG"

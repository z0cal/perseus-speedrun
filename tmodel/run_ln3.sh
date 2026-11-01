#!/bin/bash
# Record 4 with the LayerNorm inverse-sqrt seeds refit at 1e-3 (configs r4_ln3): merged capture, full plan (record
# recipe, ILP), compile_check + route_mix, compiled programs saved for tmodel/predict.py. Waits for the msobj run.
#   bash tmodel/run_ln3.sh        (log: tmodel/run_ln3.log)
set -euo pipefail
S=~/pesquisa/perseus-speedrun
R4=$S/work/perseus-r4
LOG=$S/tmodel/run_ln3.log
until grep -q '^EXIT=' "$S/tmodel/run_msobj.log" 2>/dev/null; do sleep 30; done
cd "$R4"
export PERSEUS_DATA=$S/work/data HF_HOME=$S/work/hf PYTHON=$R4/.venv/bin/python
echo "[$(date +%H:%M:%S)] merged capture" >> "$LOG"
systemd-run --user --scope -p MemoryMax=7G --quiet env BASE=graphs/r4_ln3 PLAN=bootstrap_placements/capplan_r4_ln3 JOBS=1 \
    bash speedrun/make_decode_capture.sh graphs/r4_ln3m >> "$LOG" 2>&1
C=$S/tmodel/graphs_combo/r4_ln3
mkdir -p "$C"
for b in $(seq 0 12); do ln -sfn "$R4/graphs/r4_ln3m/block_$b" "$C/block_$b"; done
ln -sfn "$R4/graphs/gpt2_decode_python_n32_kc16r_cm/block_13" "$C/block_13"
cp graphs/r4_ln3m/capture_config.json graphs/r4_ln3m/capture_env.json "$C/"
echo "[$(date +%H:%M:%S)] plan" >> "$LOG"
cd "$S"
systemd-run --user --scope -p MemoryMax=6G --quiet env MODE=base PLACER=ilp GR=graphs/r4_ln3m \
    CONFIGS=configs/model/approximation/r4_ln3/configs.json bash subagent/r4/plan_n32a4.sh r4_ln3 >> "$LOG" 2>&1
cd "$R4"
export PYTHONPATH=python CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
echo "[$(date +%H:%M:%S)] compile_check" >> "$LOG"
"$PYTHON" "$S/subagent/audit/compile_check.py" "$C" bootstrap_placements/r4_ln3_cm $(printf 'block_%d ' $(seq 0 13)) 2>&1 | tail -16 >> "$LOG"
echo "[$(date +%H:%M:%S)] route_mix" >> "$LOG"
"$PYTHON" "$S/subagent/r4/route_mix_r4.py" "$C" bootstrap_placements/r4_ln3_cm >> "$LOG" 2>&1 || true
echo "[$(date +%H:%M:%S)] done" >> "$LOG"

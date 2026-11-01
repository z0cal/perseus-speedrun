#!/bin/bash
# Runpod A/B of 2026-10-30 (tmodel): setup + smoke, then 3 decode pairs r4_ln3_sbd27d1_cm vs record 4, then 3 pairs
# sbd27d1_cm vs record 4 (server.py's DECODE_PLAN switched in the entry checkout). Outside the harness (NOSANDBOX=1).
export NOSANDBOX=1 BUNDLE=/root/perseus-entry/perseus-tmodel-entry.bundle BRANCH=tmodel-gpu CARGO_BUILD_JOBS=16 PC_NVCC_JOBS=8
cd /root/perseus-entry
L=out-ab/run_gpu.log
bash gpu/run_gpu2.sh setup && bash gpu/run_gpu2.sh smoke; echo "GOA_EXIT=$?" >> $L
PAIRS=3 bash gpu/run_gpu2.sh decode; echo "GOB_EXIT=$?" >> $L
sed -i 's|^DECODE_PLAN = "bootstrap_placements/r4_ln3_sbd27d1_cm"|DECODE_PLAN = "bootstrap_placements/sbd27d1_cm"|' perseus/speedrun/server.py
grep -n "^DECODE_PLAN" perseus/speedrun/server.py >> $L
mkdir -p out-ab-sb
OUT=/root/perseus-entry/out-ab-sb PAIRS=3 bash gpu/run_gpu2.sh decode; echo "GOC_EXIT=$?" >> $L
echo "ALL_DONE" >> $L

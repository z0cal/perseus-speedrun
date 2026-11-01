#!/bin/bash
# variant A/B: the entry checkout serves lnd_fin_e3_d27sb_cm (d27 on the sparse routes only) after go2 is done
cd /root/perseus-entry
until grep -q "GO2_EXIT" out-ab/run_gpu.log; do sleep 30; done
sed -i "s|^DECODE_PLAN = \"bootstrap_placements/lnd_fin_e3_em9_cm\"|DECODE_PLAN = \"bootstrap_placements/lnd_fin_e3_d27sb_cm\"|" perseus/speedrun/server.py
grep -n "^DECODE_PLAN" perseus/speedrun/server.py >> out-ab/run_gpu.log
export NOSANDBOX=1 OUT=/root/perseus-entry/out-ab-d27sb PAIRS=3
mkdir -p "$OUT"
bash gpu/run_gpu.sh decode; e=$?
echo "GO3_EXIT=$e" >> /root/perseus-entry/out-ab/run_gpu.log

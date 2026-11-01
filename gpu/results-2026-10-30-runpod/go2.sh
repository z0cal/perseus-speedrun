#!/bin/bash
export NOSANDBOX=1 BUNDLE=/root/perseus-entry/perseus-lndefer-entry.bundle
cd /root/perseus-entry
bash gpu/run_gpu.sh decode; e1=$?
bash gpu/run_gpu.sh prefill; e2=$?
echo "GO2_EXIT=$e1/$e2" >> out-ab/run_gpu.log

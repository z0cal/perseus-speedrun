#!/bin/bash
export NOSANDBOX=1 BUNDLE=/root/perseus-entry/perseus-lndefer-entry.bundle CARGO_BUILD_JOBS=24 PC_NVCC_JOBS=12
cd /root/perseus-entry
bash gpu/run_gpu.sh setup && bash gpu/run_gpu.sh bts && bash gpu/run_gpu.sh smoke
echo "GO1_EXIT=$?" >> out-ab/run_gpu.log

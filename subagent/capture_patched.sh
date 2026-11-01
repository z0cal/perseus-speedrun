#!/bin/bash
# capture_rec.sh through subagent/patch/run_capture.py (PATCHES=... applied to perseus.impl.attention).
#   PATCHES=defer_norm [DEFER_LAYERS=..] capture_patched.sh CAPTURE_DIR CONFIGS_JSON(absolute)
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
cd "$W/perseus-rec"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf CUDA_VISIBLE_DEVICES=
source scripts/local_env.sh
RT=$(sed -n 's/^runtime=//p' bootstrap_placements/gpt2_decode_python_n32_m16i/PLAN_CMD.txt | tr ' ' '\n' | grep = | sed 's/^/--set /')
export DEFER_CONFIGS="$2"
# shellcheck disable=SC2086
"$PYTHON" ~/pesquisa/perseus-speedrun/subagent/patch/run_capture.py --fake --tokens 1 --argmax --capture "$1" \
    --configs "$2" $RT --set SM_CHEB_GS=1

#!/bin/bash
# r4: position-0 capture in work/perseus-r4 (record 4 build, switch off: captures stay switch-free as in our pipeline),
# as the record's capture= line: run_decode --fake --tokens 1 --argmax --capture with m16i's runtime + SM_CHEB_GS=1.
#   cap0.sh OUT_GRAPH_DIR CONFIGS_JSON
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
cd "$W/perseus-r4"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf CUDA_VISIBLE_DEVICES= PYTHON=$W/perseus-r4/.venv/bin/python
source scripts/local_env.sh
RT=$(sed -n 's/^runtime=//p' bootstrap_placements/gpt2_decode_python_n32_m16i/PLAN_CMD.txt | tr ' ' '\n' | grep = | sed 's/^/--set /')
# shellcheck disable=SC2086
"$PYTHON" -m perseus.examples.gpt2.run_decode --fake --tokens 1 --argmax --capture "$1" --configs "$2" $RT --set SM_CHEB_GS=1

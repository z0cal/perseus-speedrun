#!/bin/bash
# Oracle + position-0 capture of the Decode-8 record configuration (kc16) in the record worktree.
set -euo pipefail
W=~/pesquisa/perseus-speedrun/work
cd "$W/perseus-rec"
export PERSEUS_DATA=$W/data HF_HOME=$W/hf CUDA_VISIBLE_DEVICES=
source scripts/local_env.sh
POOL=$HF_HOME/perseus/pools/openwebtext_gpt2_2000000.npy
[ -f "$ALL_BLOCKS_IO_DIR/all_blocks_L00_T128.json" ] || [ -n "$(ls -A "$ALL_BLOCKS_IO_DIR" 2>/dev/null)" ] || \
    "$PYTHON" scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 --pool "$POOL" --out "$ALL_BLOCKS_IO_DIR"
RT=$(sed -n 's/^runtime=//p' bootstrap_placements/gpt2_decode_python_n32_m16i/PLAN_CMD.txt | tr ' ' '\n' | grep = | sed 's/^/--set /')
# shellcheck disable=SC2086
"$PYTHON" -m perseus.examples.gpt2.run_decode --fake --tokens 1 --argmax --capture "${1:-$W/cap_rec_kc16}" \
    --configs "${2:-configs/model/approximation/gpt2_base_n32_kc16/configs.json}" $RT --set SM_CHEB_GS=1

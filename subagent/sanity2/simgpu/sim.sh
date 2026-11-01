#!/bin/bash
# sanity2: run_one/track do gpu/run_gpu.sh com um entry.sh falso que FALHA (rc 1) sem criar run, ou cria run e falha.
set -euo pipefail
RUNS=$PWD/runs; mkdir -p "$RUNS"; OUT=$PWD/out; mkdir -p "$OUT"
log() { echo "[log] $*"; }
fake_entry() {   # MODE=norun: fails before run.py makes a dir; MODE=void: makes a run dir with a void run.json, rc 1
    if [ "$MODE" = norun ]; then echo "run.py: error: ..." >&2; return 1; fi
    d="$RUNS/$(date +%s%N)"; mkdir -p "$d"; echo '{"void":"server exited"}' > "$d/run.json"; return 1
}
run_one() {
    local who=$1 track=$2 dir before after
    before=$(ls -1d "$RUNS"/*/ 2>/dev/null | sort)
    fake_entry >> "$OUT/${track}_$who.log" 2>&1
    after=$(comm -13 <(echo "$before") <(ls -1d "$RUNS"/*/ | sort) | tail -1)
    echo "${after%/}/run.json"
}
for MODE in norun void; do
    export MODE
    if j=$(run_one entry decode); then echo "MODE=$MODE: failure NOT detected, j='$j'"; else echo "MODE=$MODE: failure detected"; fi
    j=$(run_one entry decode) || { echo "MODE=$MODE (|| form): failure detected"; continue; }
    echo "MODE=$MODE (|| form as in run_gpu.sh): failure NOT detected, j='$j'"
done

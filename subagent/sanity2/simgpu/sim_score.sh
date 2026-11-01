#!/bin/bash
# sanity2: 'track prefill 1' -> score.py com 1 par; o que o set -euo pipefail faz com o '| tee'
set -euo pipefail
H=$HOME/pesquisa/perseus-speedrun/work/perseus-r4/speedrun/harness
mk() { python3 - "$1" "$2" <<'PY'
import json, sys
t = float(sys.argv[2])
r = {"run": sys.argv[1], "track": "prefill", "void": None, "result": {"prompts_sha256": "x", "requests": [
     {"warmup": False, "score": t, "kl_scored_mean": 1e-4, "argmax_ok": [True] * 16, "prompt": 0}]}}
json.dump(r, open(sys.argv[1] + ".json", "w"))
PY
}
mk e 10.0; mk r 10.2
python3 "$H/score.py" --track prefill --entry e.json --record r.json --json s.json | tee s.txt
echo "reached the line after score (NOT expected under set -e if score failed)"

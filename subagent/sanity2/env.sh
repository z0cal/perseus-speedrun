# source me: ambiente do Perseus na CPU (sanity2)
export R4=$HOME/pesquisa/perseus-speedrun/work/perseus-r4
export S2=$HOME/pesquisa/perseus-speedrun/subagent/sanity2
export PERSEUS_DATA=$HOME/pesquisa/perseus-speedrun/work/data
export HF_HOME=$HOME/pesquisa/perseus-speedrun/work/hf
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$R4/python
export PYTHON=$R4/.venv/bin/python
export CARGO_BUILD_JOBS=4
export TMPDIR=$S2/tmp
mkdir -p "$TMPDIR"
# heavy: prefixo para jobs pesados
HEAVY="systemd-run --user --scope -q -p MemoryMax=6G nice -n 10"

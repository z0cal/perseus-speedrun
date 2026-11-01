"""The Decode-8 candidate config from a softmax split file: the record's gpt2_base_n32_kc16 with sm_split.py's split
and round-0 bands (search.apply_split), then the candidate's other changes as cand_dfr13_cm has them: exp series
relative minimax with one squaring fewer (search2 smrel=0), GELU one BSGS level shallower (search2 gelubs), the last
normalisation deferred in layers 1 and 3 (defer_norm), CutMax with scaled normalisers (gpt2_base_n32_kc16_cm's).

    python -I build_cfg.py NAME SPLIT_JSON
Writes cand_NAME/configs.json in both worktrees' approximation configs and in the dossier's configs/."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, f"{D}/subagent")
import search  # noqa: E402
import search2  # noqa: E402

APPROX = "python/perseus/configs/model/approximation"
DEFER = (1, 3)


def build(split):
    c = search.build([])
    search.apply_split(c, split)
    search2.apply_smrel(c, 0.0)
    search2.apply_gelu(c, "fit")
    for l in DEFER:
        c["softmax"][f"transformer.h.{l}.attn"]["defer_norm"] = True
    c["cutmax"] = json.load(open(f"{search.REC}/{APPROX}/gpt2_base_n32_kc16_cm/configs.json"))["cutmax"]
    return c


def main():
    name, split = sys.argv[1], sys.argv[2]
    c = build(split)
    for d in (f"{search.W}/perseus-rec/{APPROX}", f"{search.W}/perseus/{APPROX}", f"{D}/configs"):
        os.makedirs(f"{d}/cand_{name}", exist_ok=True)
        json.dump(c, open(f"{d}/cand_{name}/configs.json", "w"), indent=1)
    print(f"cand_{name} from {os.path.basename(split)}")


if __name__ == "__main__":
    main()

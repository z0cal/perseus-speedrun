"""r4: record 4's gpt2_base_n32_kc16r_cm + our approximation knobs, for a position-0 capture and the KL check.
  smrel  search2.apply_smrel(c, 0.0): one squaring fewer per layer (r = 1; l1 >= 2 everywhere in kc16r), the exp
         series refit relative-minimax at degree 14 (BSGS depth 4 = the record's degree 8) on the 2x wider interval.
         In the CENTRED layers (ref_keys) the record normalises the series to 1 at its top (fit_centred: x exp(-delta0/2))
         so the exponentials stay <= 1 and round 0's band [exp(-U/delta2)/16, 1.05] holds: the refit is scaled by
         exp(-A) (A = the new half-width; a relative fit is scale-invariant).
  gelubs search2.apply_gelu(c, "fit") (softgelu sections are record 3's in kc16r: identical).
    python build_r4cfg.py OUT_JSON smrel gelubs"""
import json, math, os, sys
sys.path.insert(0, os.path.expanduser("~/pesquisa/perseus-speedrun/subagent"))
import numpy as np
import search2  # noqa: E402

BASE = os.path.expanduser("~/pesquisa/perseus-speedrun/work/perseus-r4/python/perseus/configs/model/approximation/"
                          "gpt2_base_n32_kc16r_cm/configs.json")
out, knobs = sys.argv[1], sys.argv[2:]
c = json.load(open(BASE))
if "smrel" in knobs:
    search2.apply_smrel(c, 0.0)
    for l in range(12):
        s = c["softmax"][f"transformer.h.{l}.attn"]
        if s.get("ref_keys"):
            A = float(s["cheb_b"])
            s["cheb_coeffs"] = list(np.asarray(s["cheb_coeffs"]) * math.exp(-A))
            s["note"] += " | centred: x exp(-A), 1 at the top (r4)"
        s["poly_coeffs"] = np.polynomial.chebyshev.cheb2poly(np.asarray(s["cheb_coeffs"])).tolist()
if "gelubs" in knobs:
    search2.apply_gelu(c, "fit")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
json.dump(c, open(out, "w"), indent=1)
print(out, knobs)

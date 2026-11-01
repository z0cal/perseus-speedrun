"""Trace the planned levels of the softmax step of a record-4 block: per node, op, planned level, refresh marks."""
import json, sys
UP = sys.argv[1]; b = int(sys.argv[2]); steps = sys.argv[3].split(",") if len(sys.argv) > 3 else ["softmax"]
g = json.load(open(f"{UP}/graphs/gpt2_decode_python_n32_kc16rm/block_{b}/graph.json"))["nodes"]
p = json.load(open(f"{UP}/bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm/block_{b}_placement.json"))
L = p["summary"]["final_named_levels"]; ss = p["sparse_slots"]; cf = p["correction_factor"]; rd = p["raise_drop"]
refr = set(cf) | set(ss) | {q["target_var"] for q in p["placements"]}
for n in g:
    st = n["step"].split(".")[-1]
    if st not in steps: continue
    o = n["output"]
    mark = ""
    if o in refr:
        mark = f"  <== REFRESH {'SB512' if ss.get(o)==512 else 'SPRU' if ss.get(o)==1 else 'dense'} drop {rd.get(o,0)}"
    print(f"{st:10s} {n['op_type']:14s} {o:7s} in={','.join(n['inputs'])[:40]:40s} lvl={L.get(o)} cap={n['output_level']} max={n.get('output_max_abs',0):.3g}{mark}")

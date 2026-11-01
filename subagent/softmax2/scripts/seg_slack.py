"""Record-4 plan: for every refresh, the planned input level (from its inputs' planned levels + the op's cost) and
landing level, so a segment's slack = cap - input level. Ops that consume a level (2 limbs): mult, square, mult by
pt/const; others keep max(in)."""
import json, sys
UP = sys.argv[1]
G = f"{UP}/graphs/gpt2_decode_python_n32_kc16rm"; P = f"{UP}/bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm"
for b in range(12):
    g = json.load(open(f"{G}/block_{b}/graph.json"))["nodes"]
    p = json.load(open(f"{P}/block_{b}_placement.json"))
    L = p["summary"]["final_named_levels"]; ss = p["sparse_slots"]; rd = p["raise_drop"]
    refr = set(p["correction_factor"]) | set(ss) | {q["target_var"] for q in p["placements"]}
    out = []
    for n in g:
        o = n["output"]
        if o not in refr: continue
        ins = [L.get(i) for i in n["inputs"] if L.get(i) is not None]
        lv = max(ins) if ins else None
        cost = 2 if n["op_type"] in ("mult", "square") else 0
        route = 'SB' if ss.get(o) == 512 else 'SPRU' if ss.get(o) == 1 else 'D'
        out.append(f"{n['step'].split('.')[-1]}:{n['op_type']}:{route} in~{(lv + cost) if lv is not None else '?'} land {L.get(o)} drop {rd.get(o, 0)}")
    print(b, " | ".join(out))

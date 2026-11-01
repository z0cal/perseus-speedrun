"""sanity2: difflib dos fluxos do token 0, gemeo d27 (servidor) x gemeo sem o switch (= captura lnd_on_e3m)."""
import collections, difflib, json, sys
A, B = sys.argv[1], sys.argv[2]          # dirs with block_N/graph.json
for b in range(14):
    try:
        na = json.load(open(f"{A}/block_{b}/graph.json"))["nodes"]
        nb = json.load(open(f"{B}/block_{b}/graph.json"))["nodes"]
    except FileNotFoundError:
        continue
    ka = [(n["op_type"], n.get("step", "")) for n in na]
    kb = [(n["op_type"], n.get("step", "")) for n in nb]
    sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
    ins, dele = collections.Counter(), collections.Counter()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for k in ka[i1:i2]:
                dele[k[0]] += 1
        if tag in ("replace", "insert"):
            for k in kb[j1:j2]:
                ins[k[0]] += 1
    oa = collections.Counter(n["op_type"] for n in na); ob = collections.Counter(n["op_type"] for n in nb)
    diffop = {k: (oa[k], ob[k]) for k in set(oa) | set(ob) if oa[k] != ob[k]}
    # names: does output v_N denote the same op in both?
    byname_a = {n["output"]: (n["op_type"], n.get("step")) for n in na}
    byname_b = {n["output"]: (n["op_type"], n.get("step")) for n in nb}
    same = sum(1 for k in byname_a if k in byname_b and byname_a[k] == byname_b[k])
    print(f"block {b}: {len(na)} (d27 twin) vs {len(nb)} (capture); op counts that differ (d27, capture): {diffop}; "
          f"names with the same op/step in both: {same}/{len(byname_b)}")
    # hint firing
    ha = [n for n in na if n["op_type"] == "hint"]; hb = [n for n in nb if n["op_type"] == "hint"]
    fa = sum(1 for n in ha if n.get("args", {}).get("fired")); fb = sum(1 for n in hb if n.get("args", {}).get("fired"))
    print(f"   hints: d27 twin {len(ha)} ({fa} fired), capture {len(hb)} ({fb} fired); "
          f"deliberate d27 {oa['deliberate_bootstrap']} capture {ob['deliberate_bootstrap']}; auto d27 {oa['auto_bootstrap']} capture {ob['auto_bootstrap']}")
    shown = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal" and shown < 3:
            print(f"   {tag}: d27[{i1}:{i2}] {[ (n['op_type'], n['output'], n.get('step')) for n in na[i1:min(i2,i1+4)]]}  capture[{j1}:{j2}] {[ (n['op_type'], n['output'], n.get('step')) for n in nb[j1:min(j2,j1+4)]]}")
            shown += 1

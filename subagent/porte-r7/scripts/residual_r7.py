"""O fluxo residual do recorde 7: em cada bloco, os níveis planejados das duas entradas de cada soma residual
(x_mid = out_proj + x_in; x_out = down + x_mid). Quando o ramo (out_proj ou down) já está no nível do fluxo, uma economia
de nível no ramo some na soma (o nível da saída é o máximo das entradas).
Uso: python3 -I residual_r7.py PROGS GRAPHS_GNABM"""
import json, sys
progs, gd = sys.argv[1:3]
print("bloco | x_mid = out_proj + x_in      | x_out = down + x_mid")
for b in range(12):
    d = json.load(open(f"{progs}/block_{b}.json"))
    g = json.load(open(f"{gd}/block_{b}/graph.json"))["nodes"]
    V, N = d["vars"], d["names"]
    def lv(nm):
        ids = [i for i, v in enumerate(V) if v["name"] == nm or v["name"].startswith(nm + "~")]
        return max(V[i]["meta"]["level"] for i in ids) if ids else None
    adds = [n for n in g if n["step"] == f"blk{b}" and n["op_type"] == "add"]
    row = []
    for n in adds:
        a, c = n["inputs"]
        row.append(f"{lv(n['output'])} = max({lv(a)}, {lv(c)})")
    print(f"{b:5d} | {row[0]:28s} | {row[1] if len(row) > 1 else '-'}")

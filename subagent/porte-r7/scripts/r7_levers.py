"""Modelo R7 (tmodel/units_r7.py + r7_estimate.py) aplicado às alavancas das 4 ideias, sobre os programas compilados do
recorde 7 (_gnabd7). Diferencial, sem GPU. Imprime:
 1. custo de cada refresh por rota (o SparseB-256 o modelo precifica como 512: limite superior);
 2. valor de +1 raise drop em cada SparseB-256 (refresh mais barato + ops do trecho seguinte 1 nível mais fundo);
 3. custo de uma economia de nível NÃO monetizada (ops a jusante 1 nível mais raso: mais limbs) por trecho;
 4. custo de um produto ct x ct extra por nível (para os produtos que cada ideia acrescenta).
Uso: python3 -I r7_levers.py TMODEL_DIR PROGS_DIR GRAPHS_GNABM"""
import collections, json, sys
TM, PROGS, GD = sys.argv[1:4]
sys.path.insert(0, TM)
import r7_estimate as E  # noqa: E402
import units as U  # noqa: E402
import units_r7 as R  # noqa: E402
import tokplan as T  # noqa: E402
T.NQ = E.NQ
SCALE = 968.0 / 1086.0     # fator do modelo R7 (absoluto 1086 contra 968 medidos)


def kind(o):
    return o["kind"] if isinstance(o["kind"], str) else next(iter(o["kind"]))


def site_ms(x):
    if x["route"] == "1":
        return E.SPRU_R7
    site = dict(x, route="512" if x["route"] == "256" else x["route"])
    T.em_list = lambda top, series, k24: []
    u = T.site_units(site, "shipped", "shipped")
    chains = 1 if (x["real"] or x["route"] != "stcf") else 2
    U.add(u, R.evalmod(E.EMR["Shipped_K16"][E.em_top(x)]["oplist"]), chains)
    return E.ms(u)


def op_ms(o, lv):
    """absolute_r7's per-op pricing, lv: var index -> primes dropped."""
    acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
    def one(o):
        k = kind(o)
        if k == "Batched":
            for x in o["kind"]["Batched"]:
                one(x)
            return
        if k == "Bootstrap":
            return
        if k == "MultRescale":
            ins = [lv(i) for i in o["ins"]]
            outs = [lv(i) for i in o["outs"]]
            n = E.NQ - max(ins)
            drop = max(outs) - max(ins)
            for pk in o["pre"]:
                if (next(iter(pk)) if isinstance(pk, dict) else pk) == "Rescale":
                    U.add(acc["rescale"], U.rescale(n + 2))
            if len(o["ins"]) == 2 or k == "Square":
                U.add(acc["ks"], R.relin(n, fused=True, drop=max(drop, 1)))
            else:
                U.add(acc["ew"], [0, 0, 3 * n])
                U.add(acc["rescale"], U.rescale(n))
            return
        T.op_units(o, lv, acc)
    one(o)
    return sum(E.ms(acc[c]) for c in ("ks", "rescale", "ew"))


def load(b):
    d = json.load(open(f"{PROGS}/block_{b}.json"))
    g = json.load(open(f"{GD}/block_{b}/graph.json"))["nodes"] if b < 13 else None
    return d, g


def lvfun(V, shift=None, delta=0):
    def lv(i):
        L = V[i]["meta"]["level"]
        if shift is not None and i in shift:
            L = max(L + delta, 0)
            while L not in E.DROP:      # n32sp's grid has holes (31, 35, ...): the next level that exists
                L += 1 if delta > 0 else -1
        return E.DROP[L]
    return lv


def forward(d, start_vars, stop_at_bootstrap=True):
    """Vars reachable from start_vars through the program's ops, not crossing a Bootstrap op."""
    reach = set(start_vars)
    ops = []
    for o in d["ops"]:
        k = kind(o)
        ins = set(o["ins"])
        if k == "Batched":
            ins = set(i for x in o["kind"]["Batched"] for i in x["ins"])
        if ins & reach:
            if k == "Bootstrap" and stop_at_bootstrap:
                continue
            outs = o["outs"] if k != "Batched" else [i for x in o["kind"]["Batched"] for i in x["outs"]]
            reach |= set(outs)
            ops.append(o)
    return reach, ops


def var_ids(d, name):
    """A graph name's program vars: itself and its realized forms (v~rr, v~er, v~rs: the fused / eager rescales)."""
    return [i for i, v in enumerate(d["vars"]) if v["name"] == name or v["name"].startswith(name + "~")]


def seg_shift_ms(d, start_vars, delta):
    """Cost change when start_vars move `delta` grid units: each op's outputs keep their consumption over the op's
    deepest input (out - max(in)), so a merge with an unshifted deeper input absorbs the shift; refreshes stop it."""
    V = d["vars"]
    new = {i: V[i]["meta"]["level"] + delta for i in start_vars}
    old_lv = lvfun(V)
    def nl(i):
        return new.get(i, V[i]["meta"]["level"])
    def snap(L, ref):
        L = max(L, 0)
        while L not in E.DROP:
            L += 1 if L < ref else -1
        return L
    dm, n = 0.0, 0
    def subs(o):
        return o["kind"]["Batched"] if kind(o) == "Batched" else [o]
    for o in d["ops"]:
        if kind(o) == "Bootstrap":
            continue
        for x in subs(o):
            ins = x["ins"]
            if not ins or all(nl(i) == V[i]["meta"]["level"] for i in ins):
                continue
            mi_old = max(V[i]["meta"]["level"] for i in ins)
            mi_new = max(nl(i) for i in ins)
            if mi_new == mi_old and all(nl(i) == V[i]["meta"]["level"] for i in x["outs"]):
                pass
            for j in x["outs"]:
                new[j] = snap(V[j]["meta"]["level"] + (mi_new - mi_old), V[j]["meta"]["level"])
            lvn = lambda i: E.DROP[snap(nl(i), V[i]["meta"]["level"])]  # noqa: E731
            c0, c1 = op_ms(x, old_lv), op_ms(x, lvn)
            if c0 != c1:
                n += 1
            dm += c1 - c0
    return dm, n


def bts_ops(d):
    return [o for o in d["ops"] if kind(o) == "Bootstrap"]


def main():
    ss = E.sites(PROGS)
    by = collections.defaultdict(list)
    for x in ss:
        by[(x["route"] + ("r" if x["real"] else "c"), x["drop"])].append(site_ms(x))
    print("1. custo por refresh no modelo R7 (ms, bruto; x0.89 = escalado)")
    for k in sorted(by):
        v = by[k]
        print(f"   {k[0]:7s} drop {k[1]}: n={len(v):2d}  {sum(v) / len(v):6.2f} ms ({sum(v) / len(v) * SCALE:5.2f} escalado)")
    tot = sum(site_ms(x) for x in ss)
    print(f"   total refreshes {tot:.0f} ms/token")

    print("\n2. +1 raise drop num SparseB-256 (bruto): refresh / ops do trecho seguinte 1 nível mais fundo / soma")
    gains = []
    for b in range(12):
        d, g = load(b)
        bo = [o for o in bts_ops(d)]
        sites = d["bts_sites"]
        for o, s in zip(bo, sites):
            r = s["route"]
            if r["slots"] != 256:
                continue
            x = dict(block=b, route="256", real=r["real"], drop=r["raise_drop"], lout=s["landing"]["level"])
            if x["drop"] >= 3:
                continue
            x2 = dict(x, drop=x["drop"] + 1, lout=x["lout"] + 2)
            dr = site_ms(x2) - site_ms(x)
            dops, n = seg_shift_ms(d, o["outs"], +2)
            gains.append((b, s["var"], x["drop"], dr, dops))
            print(f"   bloco {b:2d} {s['var']:7s} drop {x['drop']}->{x['drop'] + 1}: refresh {dr:+.2f}  ops ({n} ops) {dops:+.2f}  = {dr + dops:+.2f} ms")
    print(f"   média por drop: {sum(a[3] + a[4] for a in gains) / len(gains):+.2f} ms bruto ({sum(a[3] + a[4] for a in gains) / len(gains) * SCALE:+.2f} escalado); "
          f"todos os {len(gains)} SB-256 com drop < 3: {sum(a[3] + a[4] for a in gains):+.1f} ms")

    print("\n3. economia de 1 nível NÃO monetizada (ops a jusante 1 nível mais raso), por trecho e passo de origem (bruto)")
    acc = collections.defaultdict(float)
    for b in range(12):
        d, g = load(b)
        names = d["names"]
        step_of = {n["output"]: n["step"] for n in g}
        V = d["vars"]
        # ln_2 / ln_1: o produto vetor x 1/sigma (o último 'mult' ct x ct do passo antes do add_pt do shift)
        for st in ("ln_1", "ln_2"):
            prods = [n for n in g if n["step"] == f"blk{b}.{st}" and n["op_type"] == "mult"
                     and all(x.startswith("v_") for x in n["inputs"])]
            if not prods:
                continue
            v = prods[-1]["output"]
            dm, n = seg_shift_ms(d, var_ids(d, v), -2)
            acc[f"LN adiada: {st} (ops depois do produto)"] += dm
        # softmax: a saída final (o último var do passo softmax) -> softmax_v, out_proj, ln_2, up
        sm = [n for n in g if n["step"] == f"blk{b}.softmax"]
        v = sm[-1]["output"]
        dm, n = seg_shift_ms(d, var_ids(d, v), -2)
        acc["normalização adiada: ops depois da saída da softmax"] += dm
        # smrel: as quadraturas do exp logo depois do SB#1 (o pouso do SB#1 e tudo até o SB#2)
        bo = bts_ops(d)
        sb = [(o, s) for o, s in zip(bo, d["bts_sites"]) if s["route"]["slots"] == 256]
        if sb:
            dm, n = seg_shift_ms(d, sb[0][0]["outs"], -2)
            acc["smrel: trecho C inteiro 1 nível mais raso (sem drop)"] += dm
    for k, v in acc.items():
        print(f"   {k:58s} {v:+6.1f} ms/token bruto ({v * SCALE:+5.1f} escalado)")

    print("\n4. um produto ct x ct (relin + rescale fundidos) por nível da grade (bruto, ms)")
    for L in (28, 32, 36, 40, 44, 48, 52):
        n = E.NQ - E.DROP[L]
        nn = E.NQ - E.DROP[L + 2]
        u = R.relin(n, fused=True, drop=n - nn)
        r = U.ks(n, "n32a4")
        print(f"   nível {L} ({n} limbs): produto {E.ms(u):.3f} ms, rotação {E.ms(r):.3f} ms")


if __name__ == "__main__":
    main()


def existing_slack():
    """A folga que o plano do recorde 7 JÁ tem e não usa: SB#2 com drop < 3 cujo trecho D cabe com mais drop (entrada do
    denso <= 52), e o denso pré-GELU com drop 0 (teto 1 pela guarda [user_region]) cujo trecho A cabe."""
    print("\n5. folga que o plano do recorde 7 já tem e não monetiza (drop extra que os tetos de nível permitem; bruto)")
    tot = 0.0
    for b in range(12):
        d, g = load(b)
        bo = bts_ops(d)
        sites = d["bts_sites"]
        pairs = list(zip(bo, sites))
        main = [(o, s) for o, s in pairs if s["route"]["slots"] != 1]
        sb = [(o, s) for o, s in main if s["route"]["slots"] == 256]
        dense = [(o, s) for o, s in main if s["route"]["slots"] == 0 and s["kind"] != "Deliberate"]
        if not sb or not dense:
            continue
        o2, s2 = sb[-1]
        od, sd = dense[0]
        slackD = (52 - sd["input"]["level"]) // 2
        k = min(3 - s2["route"]["raise_drop"], slackD)
        if k > 0:
            x = dict(block=b, route="256", real=False, drop=s2["route"]["raise_drop"], lout=s2["landing"]["level"])
            x2 = dict(x, drop=x["drop"] + k, lout=x["lout"] + 2 * k)
            dr = site_ms(x2) - site_ms(x)
            dops, n = seg_shift_ms(d, o2["outs"], +2 * k)
            tot += dr + dops
            print(f"   bloco {b:2d} SB#2 {s2['var']:6s} drop {x['drop']}->{x['drop'] + k} (folga D {slackD} níveis): {dr + dops:+.2f} ms")
    print(f"   total: {tot:+.1f} ms/token bruto ({tot * SCALE:+.1f} escalado) — sem nenhuma das 4 ideias")


if __name__ == "__main__":
    existing_slack()


def smrel_net():
    """smrel no recorde 7 (smrel_r7.py): exp grau 6 -> 10-11 (profundidade 3 -> 4, +2 produtos ct x ct no trecho B),
    uma quadratura a menos logo depois do SB#1 (-1 produto e -1 nível no trecho C). Melhor caso: o drop search dá +1 drop
    ao SB#1 onde o drop < 3 (o trecho C fica com os mesmos níveis); onde o SB#1 já tem drop 3, o trecho C fica 1 nível
    mais raso (ops mais caras)."""
    print("\n6. smrel no recorde 7, por bloco (bruto): SB#1 +1 drop onde dá, senão o trecho C 1 nível mais raso; "
          "+2 produtos no B, -1 quadratura no C")
    tot_best, tot_nodrop = 0.0, 0.0
    for b in range(12):
        d, g = load(b)
        pairs = [(o, s) for o, s in zip(bts_ops(d), d["bts_sites"]) if s["route"]["slots"] == 256]
        o1, s1 = pairs[0]
        L_in = s1["input"]["level"]
        n_b = E.NQ - E.DROP[L_in - 2]         # the two extra products: at the series' last levels, before SB#1
        prod_b = 2 * E.ms(R.relin(n_b, fused=True, drop=(E.DROP[L_in] - E.DROP[L_in - 2])))
        Lc = s1["landing"]["level"]
        n_c = E.NQ - E.DROP[Lc]
        sq_c = E.ms(R.relin(n_c, fused=True, drop=(E.DROP[Lc + 2] - E.DROP[Lc])))
        shallow, _ = seg_shift_ms(d, o1["outs"], -2)
        if s1["route"]["raise_drop"] < 3:
            x = dict(block=b, route="256", real=False, drop=s1["route"]["raise_drop"], lout=Lc)
            gain = site_ms(dict(x, drop=x["drop"] + 1, lout=Lc + 2)) - site_ms(x)
            best = prod_b - sq_c + gain
            how = f"SB#1 drop {x['drop']}->{x['drop'] + 1} {gain:+.2f}"
        else:
            best = prod_b - sq_c + shallow
            how = f"SB#1 já em drop 3: C mais raso {shallow:+.2f}"
        nod = prod_b - sq_c + shallow
        tot_best += best; tot_nodrop += nod
        print(f"   bloco {b:2d}: +2 produtos {prod_b:+.2f}, -1 quadratura {-sq_c:+.2f}, {how} -> {best:+.2f} ms "
              f"(sem drop: {nod:+.2f})")
    print(f"   total: melhor caso {tot_best:+.1f} ms bruto ({tot_best * SCALE:+.1f} escalado); se o drop search não usar a "
          f"folga: {tot_nodrop:+.1f} ms ({tot_nodrop * SCALE:+.1f})")


def defer_norm_cost():
    """Normalização adiada com pair_heads: +2 produtos ct x ct (um por deslocamento do unpack) e +1 rotação de R por
    camada, no nível da saída da softmax; o produto z R do gs_num (de graça) sai."""
    print("\n7. normalização adiada no recorde 7: produtos e rotação a mais por camada (bruto)")
    tot = 0.0
    for b in range(12):
        d, g = load(b)
        sm = [n for n in g if n["step"] == f"blk{b}.softmax"]
        ids = var_ids(d, sm[-1]["output"])
        L = max(d["vars"][i]["meta"]["level"] for i in ids)
        n = E.NQ - E.DROP[L]
        c = 2 * E.ms(R.relin(n, fused=True, drop=E.DROP[L + 2] - E.DROP[L])) + E.ms(U.ks(n, "n32a4"))
        tot += c
    print(f"   +{tot:.1f} ms/token bruto ({tot * SCALE:+.1f} escalado)")


if __name__ == "__main__":
    smrel_net()
    defer_norm_cost()

"""O modelo de tempo (units.py + calib.json) aplicado ao recorde 7 (FlippingToTokenPerSecond, n32sp): quanto o
EvalMod par (PC_EM_EVEN) e a GELU par (GeluCfg.even, configs _ge / _ge48) mudam o ms/token.

As duas mudanças são locais, então a estimativa é DIFERENCIAL sobre o plano servido `_gnabd7`:
- EvalMod: os 14 programas compilados (pc-program, exemplo save_progs) dão cada refresh (rota, real/complexo, drop,
  pouso); o topo de entrada do EvalMod sai do pouso (StC-first: o pouso é a saída do EvalMod, o StC vem antes do
  raise; SparseB: 1 estágio de StC depois); o custo de cada série é o censo de ops do builder lean
  (em/n32sp_r7.jsonl, exemplo emcost) em units.evalmod. Pousos iguais bit a bit ⇒ nada mais muda.
- GELU: a série é avaliada com ops que seguem a profundidade; profundidade d ⇒ nível da grade L0 + 2 d (L0 = nível de
  entrada da GELU no grafo do bloco) ⇒ limbs pela tabela da n32sp (levels_n32sp.json); custo como em tokplan.op_units.
  A forma do recorde (eval_chebyshev_masked, mask_fold 2) é conferida contra os produtos do grafo capturado.
usage: python -I r7_estimate.py PROGS_DIR GRAPH_DIR PERSEUS_ROOT
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import units as U  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(f"{H}/calib.json"))
CH = "n32a4"            # n32sp: 51 Q primes, 13 special, dnum 4 (dígitos de 13), como a n32a4
NQ = 51
DROP = {int(k): v for k, v in json.load(open(f"{H}/levels_n32sp.json"))["dropped"].items()}
RECORD_MS = 968.0       # recorde 7, harness (README do upstream)


def ms(u):
    return C["tau"] * u[0] + C["kappa"] * u[1] + C["eps"] * u[2]


def limbs(level):
    return NQ - DROP[level]


# ── EvalMod ──
EM = collections.defaultdict(dict)
for line in open(f"{H}/em/n32sp_r7.jsonl"):
    j = json.loads(line)
    EM[j["name"]][j["top"]] = j
OUT2IN = {j["out_top"]: t for t, j in EM["Shipped_K16"].items()}


def em_units(series, top, only=None):
    ops = EM[series][top]["oplist"]
    if only is not None:
        ops = [o for o in ops if o[0] in only]
    return U.evalmod(ops, CH)


def sites(progs):
    out = []
    for b in range(14):
        d = json.load(open(f"{progs}/block_{b}.json"))
        for s in d["bts_sites"]:
            r = s["route"]
            route = "stcf" if (r["slots"] == 0 and r["stc_first"]) else ("dense" if r["slots"] == 0 else str(r["slots"]))
            out.append(dict(block=b, route=route, real=r["real"], drop=r["raise_drop"], lout=s["landing"]["level"]))
    return out


def em_top(s):
    land_top = NQ - 1 - DROP[s["lout"]]
    if s["route"] == "stcf":
        out_top = land_top
    elif s["route"] in ("256", "512"):
        out_top = land_top + 2          # 1 StC stage after EvalMod
    else:
        return None
    return OUT2IN[out_top]


def em_delta(ss, only=None):
    tot, by = 0.0, collections.defaultdict(float)
    for s in ss:
        top = em_top(s)
        if top is None:
            continue
        chains = 1 if (s["real"] or s["route"] != "stcf") else 2
        dm = chains * (ms(em_units("Even_K16", top, only)) - ms(em_units("Shipped_K16", top, only)))
        tot += dm
        by[f"{s['route']}{'r' if s['real'] else 'c'} top{top}"] += dm
    return tot, dict(by)


# ── GELU ──
class _V:
    def __init__(self, v):
        self.d = v


class CostOps:
    """Depth-tracking ops pricing each op at its level (tokplan.op_units' units: Mult/Square = key switch + tensor 4n,
    each product's rescale, MultConst / MultPt / Add / AddConst / AddPt elementwise)."""
    cheb_bsgs = dead_ops = True
    realize_shared = cheb_align = False
    gelu_fold = True

    def __init__(self, L0):
        self.L0, self.u, self.ctct, self.lv = L0, U.z(), 0, collections.Counter()

    def n(self, d):
        return limbs(self.L0 + 2 * d)

    def mult(self, a, b):
        if isinstance(b, _V):
            d = max(a.d, b.d)
            n = self.n(d)
            U.add(self.u, U.ks(n, CH))
            U.add(self.u, [0, 0, 4 * n])
            self.ctct += 1
            self.lv[self.L0 + 2 * d] += 1
        else:
            d = a.d
            n = self.n(d)
            U.add(self.u, [0, 0, 2 * n])
        U.add(self.u, U.rescale(n))
        return _V(d + 1)

    def square(self, a):
        return self.mult(a, a)

    def add(self, a, b):
        d = max(a.d, b.d) if isinstance(b, _V) else a.d
        U.add(self.u, [0, 0, (2 if isinstance(b, _V) else 1) * self.n(d)])
        return _V(d)

    sub = add

    def inplace_add(self, a, b):
        a.d = max(a.d, b.d)
        U.add(self.u, [0, 0, 2 * self.n(a.d)])

    def mask(self, ct, s):          # MultPt + its rescale
        n = self.n(ct.d)
        U.add(self.u, [0, 0, 3 * n])
        U.add(self.u, U.rescale(n))
        return _V(ct.d + 1)

    def add_mask(self, ct, s):      # AddPt
        U.add(self.u, [0, 0, self.n(ct.d)])
        return _V(ct.d)


def gelu_cost(root, cfgj, L0, even):
    sys.path.insert(0, f"{root}/python")
    from perseus.impl.poly import eval_chebyshev_even, eval_chebyshev_masked
    o = CostOps(L0)
    c = [cfgj["xmax"] * float(v) for v in cfgj["cheb_coeffs"]]
    f = eval_chebyshev_even if even else eval_chebyshev_masked
    out = f(o, _V(0), c, cfgj["cheb_a"], cfgj["cheb_b"], cfgj["cheb_eps"], o.mask, o.add_mask, leaf_terms="budget")
    return o, out.d


def graph_gelu(gdir, progs, b):
    """The GELU's entry level and its ciphertext x ciphertext products by input level, at the PLANNED levels (the
    compiled program's vars; the graph's own levels are the capture's)."""
    nodes = json.load(open(f"{gdir}/block_{b}/graph.json"))["nodes"]
    d = json.load(open(f"{progs}/block_{b}.json"))
    def lvl(name):      # a capture's reactive refresh the plan does not keep: its input
        name = name if name in d["names"] else name.replace("_auto_bootstrapped", "")
        return d["vars"][d["names"][name]]["meta"]["level"]
    g = [n for n in nodes if n["step"] == f"blk{b}.gelu"]
    ct = lambda x: x.startswith("v_")  # noqa: E731
    L0 = lvl(g[0]["inputs"][0])
    prods = collections.Counter(max(lvl(x) for x in n["inputs"]) for n in g
                                if n["op_type"] in ("mult", "square") and all(ct(x) for x in n["inputs"]))
    return L0, prods, sum(1 for n in g if n["op_type"] == "auto_bootstrap")


def absolute(progs, ss):
    """The record's token by the model (tokplan.op_units on n32sp's level grid; MultRescale = product + rescale; the
    refreshes by tokplan.site_units with the EvalMod top from the landing; SPRU at predict.SPRU_MS)."""
    import tokplan as T
    T.NQ = NQ
    acc = {c: U.z() for c in ("ks", "rescale", "ew", "launch")}
    for b in range(14):
        d = json.load(open(f"{progs}/block_{b}.json"))
        V = d["vars"]
        lv = lambda i: DROP[V[i]["meta"]["level"]]  # noqa: E731

        def one(o):
            k = T.kind(o)
            if k == "Batched":
                for x in o["kind"]["Batched"]:
                    one(x)
                return
            if k == "MultRescale":
                o = dict(o, kind="Mult", pre=list(o["pre"]) + ["Rescale"])
            T.op_units(o, lv, acc)
        for o in d["ops"]:
            one(o)
    ops = sum(ms(acc[c]) for c in ("ks", "rescale", "ew"))
    T.em_list = lambda top, series, k24: EM["Shipped_K16"][top]["oplist"]
    ref = 0.0
    for x in ss:
        if x["route"] == "1":
            ref += 8.0
            continue
        site = dict(x, route="512" if x["route"] == "256" else x["route"])   # SparseB-256 priced as 512 (upper bound)
        u = T.site_units(site, "shipped", "shipped")
        # site_units puts EvalMod at n32a4 record 4's tops: replace by the landing's top
        top0 = (42 - 2 * (x["drop"] or 0)) if x["route"] in ("stcf", "dense") else (48 - 2 * (x["drop"] or 0))
        chains = 1 if (x["real"] or x["route"] != "stcf") else 2
        U.add(u, U.evalmod(EM["Shipped_K16"][top0]["oplist"], CH), -chains)
        U.add(u, U.evalmod(EM["Shipped_K16"][em_top(x)]["oplist"], CH), chains)
        ref += ms(u)
    return ops, ref


def main():
    progs, gdir, root = sys.argv[1:4]
    ss = sites(progs)
    mix = collections.Counter(s["route"] + ("r" if s["real"] else "c") for s in ss)
    print(f"refreshes/token {len(ss)}: {dict(mix)}")
    ops, ref = absolute(progs, ss)
    scale = RECORD_MS / (ops + ref)
    print(f"modelo do recorde 7: ops {ops:.0f} + refreshes {ref:.0f} = {ops + ref:.0f} ms/token (medido {RECORD_MS:.0f}): "
          f"fator {scale:.2f} (o modelo é do runtime do recorde 4)")
    em, by = em_delta(ss)
    em_relin, _ = em_delta(ss, only=("mult", "multacc", "wsum"))
    ship = sum((1 if (s['real'] or s['route'] != 'stcf') else 2) * ms(em_units("Shipped_K16", em_top(s)))
               for s in ss if em_top(s) is not None)
    print(f"\nEvalMod (modelo): recorde {ship:.1f} ms/token em EvalMod; par {ship + em:.1f}; delta {em:+.1f} ms "
          f"({100 * em / RECORD_MS:+.2f} % de {RECORD_MS:.0f} ms)")
    print(f"   só relin + WSum (rescales já fundidos no runtime): {em_relin:+.1f} ms ({100 * em_relin / RECORD_MS:+.2f} %)")
    for k, v in sorted(by.items()):
        print(f"   {k:16s} {v:+6.2f} ms")

    base = json.load(open(f"{root}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab/configs.json"))["softgelu"]
    print("\nGELU (modelo; níveis da forma do recorde conferidos contra os produtos ct x ct do programa compilado):")
    tot = {}
    # só zerar os ímpares, com o avaliador do recorde (sem a forma em z): quanto do ganho do _ge é a forma par
    ev = json.load(open(f"{root}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab_ge/configs.json"))["softgelu"]
    t = 0.0
    for b in range(12):
        key = f"transformer.h.{b}.mlp.act"
        L0, _, _ = graph_gelu(gdir, progs, b)
        t += ms(gelu_cost(root, ev[key], L0, False)[0].u) - ms(gelu_cost(root, base[key], L0, False)[0].u)
    tot["zero_odd"] = t
    print(f"   só zerar os ímpares (avaliador do recorde): {t:+.2f} ms/token ({100 * t / RECORD_MS:+.2f} %)")
    for var in ("ge", "ge48"):
        ev = json.load(open(f"{root}/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab_{var}/configs.json"))["softgelu"]
        t = 0.0
        for b in range(12):
            key = f"transformer.h.{b}.mlp.act"
            L0, gp, nb = graph_gelu(gdir, progs, b)
            ob, db = gelu_cost(root, base[key], L0, False)
            if ev[key].get("even"):
                oe, de = gelu_cost(root, ev[key], L0, True)
                assert de == db, (b, de, db)
            else:
                oe, de = ob, db
            dm = ms(oe.u) - ms(ob.u)
            t += dm
            if var == "ge":
                ok = "= programa" if dict(ob.lv) == dict(gp) else f"!= programa {sorted(gp.items())} (sim {sorted(ob.lv.items())})"
                print(f"   bloco {b:2d}: L0 {L0}, produtos {ob.ctct} -> {oe.ctct}, {ms(ob.u):5.2f} -> {ms(oe.u):5.2f} ms "
                      f"({dm:+.2f}); níveis dos produtos do recorde {ok}")
        tot[var] = t
        print(f"   _{var}: delta {t:+.2f} ms/token ({100 * t / RECORD_MS:+.2f} %)")
    print("\nTotal (% de 968 ms)            modelo bruto      escalado pelo fator")
    for name, v in (("EvalMod par", em), ("EvalMod par, só relin+WSum", em_relin), ("GELU _ge", tot["ge"]),
                    ("GELU _ge48", tot["ge48"]), ("GELU só ímpares zerados", tot["zero_odd"]), ("EvalMod par + GELU _ge", em + tot["ge"]),
                    ("EvalMod par + GELU _ge48", em + tot["ge48"])):
        print(f"   {name:28s} {v:+7.1f} ms {100 * v / RECORD_MS:+6.2f} %   {v * scale:+7.1f} ms {100 * v * scale / RECORD_MS:+6.2f} %")
    print("Regra 5: a entrada precisa de <= -1 %.")


if __name__ == "__main__":
    main()

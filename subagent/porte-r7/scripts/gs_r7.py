"""Goldschmidt re-dividido sobre as configs do recorde 7 (gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab): contagens por
rodada do design() servido (smcheb.design guarda o primeiro mínimo da grade) contra as divisões de MESMA soma com outro
alvo intermediário; resíduo da última rodada (fator |sum p - 1| por linha) no pior kc de 1..16 (o decode tem <= 16
chaves). Só CPU, só o smcheb do upstream. Uso: python3 -I gs_r7.py UPSTREAM_CLONE"""
import json, sys
sys.path.insert(0, sys.argv[1] + "/python")
from perseus.impl import smcheb  # noqa: E402
C = json.load(open(sys.argv[1] + "/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab/configs.json"))["softmax"]
KCS = range(1, 17)
MIDS = (0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def last_resid(m, k):
    return max(smcheb.resid(*smcheb.refine_band(kc, m), k) for kc in KCS)


for L in range(12):
    c = C[f"transformer.h.{L}.attn"]
    rec = [int(x) for x in c["recip_iters_cheb"]]
    tg = [float(x) for x in c["recip_targets"]]
    lo0, hi0 = smcheb.band_from_seed(c["init_alpha"], c["init_beta"])
    if len(rec) != 2:
        # 3+ rodadas: só o resíduo da última com o alvo servido, e a melhor troca de 1 iteração entre a 1ª e a última
        r = last_resid(tg[-2], rec[-1])
        print(f"camada {L:2d}: {rec} alvos {tg}: fator final {r:.1e} (3+ rodadas, não varrido)")
        continue
    r_rec = last_resid(tg[0], rec[1])
    alts = []
    for m in MIDS:
        k0 = smcheb.k_for(lo0, hi0, m)
        for k1 in range(1, 12):
            if k0 + k1 == sum(rec):
                alts.append((last_resid(m, k1), k0, k1, m))
    best = min(alts)
    print(f"camada {L:2d}: servido {rec} (alvo intermediário {tg[0]}) fator final {r_rec:.1e} | melhor de mesma soma "
          f"[{best[1]}, {best[2]}] (alvo {best[3]}) {best[0]:.1e}  = {r_rec / best[0]:.0f}x menor")

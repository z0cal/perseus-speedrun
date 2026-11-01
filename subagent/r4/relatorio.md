# Nossa entrada contra o recorde 4 (LeanChainDecode, 1,798 s/token) — 2026-10-28

Analista independente. Só CPU, nada de servidor. Não mexi em `work/perseus`, `work/perseus-rec`, `scripts/`,
`configs/`, `plans/`, `README.md` nem `HANDOFF.md`. Não rodei nenhum `rm`. Todo job pesado rodou um por vez, sob
`systemd-run --user --scope -p MemoryMax=5G`.

Para testar, criei um worktree novo `work/perseus-r4`: `origin/speedrun` = d97a6f7, branch local `r4-em9` (1ca6f1d e
fc19e09 = 67717aa e d8a7643 portados), venv próprio e `_pc` compilado nele. Scripts e saídas ficam em
`subagent/r4/`. Na tabela, "medido" quer dizer executado nesta sessão.

## Veredito: **NO-GO** para o em9 (EvalMod d27) sobre o recorde 4

| plano (Decode-8, cadeia n32a4, capturas do recorde) | refreshes/token | dense / SparseB-512 / SPRU | fonte |
|---|---|---|---|
| recorde 4 `gpt2_decode_python_n32a4_kc16r_cm` | **114** (101 + 13) | 42 / 40 / 32 | medido (compile) |
| recorde 4 + em9 (`r4f_em9_ilps_cm`, ILP) | **113** (100 + 13) | 42 / 39 / 32 | **medido (ILP)** |

O em9 sozinho economiza **1 refresh por token** (um SparseB-512 no bloco 4: 10 → 9). No recorde 3 o mesmo switch,
junto com a softmax, economizava 10. Com o custo do EvalMod dense mais caro, o saldo de tempo fica entre −0,1 % e
+1,2 % (mais lento). Não passa a regra 5 (≥ 1 % mais rápido).

## 1. O EvalMod de 9 níveis ainda poupa 1 nível na n32a4? — SIM (medido)

- `ChainSpec::n32a4` (`crates/pc-core/src/params.rs:104-107`) é a n32 com `thin: 3, dnum: 4`. Os grupos de primos,
  do fundo para o topo: `[2, 1, 2, 2, …, 2]` (51 limbs). Só o s1 é primo simples. Os 46 primos do topo (EvalMod,
  CtS, níveis do usuário) continuam em pares compostos.
- `subagent/r4/emscan4/` roda o `build_evalmod` (a mesma rotina do `EvalModPlan::program` da GPU) nas duas cadeias,
  para todo topo de entrada em nível composto, nl 1 e 2, K16 e K24 (`emscan4.out`):
  - n32: 0 divergências;
  - n32a4: d27 = sempre 2 primos a menos que a série shipped (dense 16/18 contra 18/20 primos; ops 98/100 contra
    78/80; K24 98/100 contra 97/99). As únicas 4 divergências ficam nos topos de entrada 18/20, onde a saída cairia
    no q0/s1. Nenhuma rota alimenta o EvalMod tão fundo (o raise vai ao topo 50).
- Os pousos do compilador continuam em níveis efetivos da n32: a receita n32a4 é a do recorde mais `MAX_LEVEL=48
  PLAN_REFRESH_ENV_CAP=48` (`speedrun/make_n32a4_decode_plan.sh`). O recorde 4 só acrescentou `input_room` em
  `route.rs` e o `n_q` da cadeia em `compile/mod.rs`. O `shift_evalmod` vale do mesmo jeito.
- Com o porte, o plano do recorde 4 compila com **0 erros**, e com `FIDESLIB_BTS_EVALMOD=d27` dá **33.601 erros**: o
  compilador enxerga os pousos novos na n32a4.

## 2. Quantos refreshes o em9 poupa no plano do recorde 4 — **1** (ILP medida)

Receita em9 = receita do recorde com `BTS_LEVEL 34→32, 28→26`, `512:26→24`, `STC_FIRST_FALLBACK 34:1→32:1`, remap
`24:14 → 26:24`. O remap é idempotente: no `kc16rm` os deliberados foram capturados em 26 (12) e 14 (6), nada em 24,
então o `24:14` do recorde é no-op e a dupla aplicação não muda nada. Fluxo: `subagent/r4/plan_n32a4.sh`, cópia do
`make_n32a4_decode_plan.sh` com `MODE`/`PLACER`, sem `rm`.

| | blocos 0-12 | CutMax (ILP) | total |
|---|---|---|---|
| recorde 4 (ILP) | 8,8,8,8,10,8,8,8,8,8,8,10,1 = 101 | 13 | 114 |
| em9, ILP (blocos 0-11; estágio 12 ver abaixo) | 8,8,8,8,**9**,8,8,8,8,8,8,10,1 = **100** | 13 | **113** |
| base, proxy min-cut (`kc16rm`) | 109 | 13 | 122 |
| em9, proxy min-cut (`kc16rm`) | 105 | 13 | 118 |
| base / em9, min-cut na **posição 0** (`kc16r`) | 113 / 104 | — | — |

Checagens do `r4f_em9_ilps_cm`:
- `compile_check` nos 14 estágios encadeados (`subagent/r4/graphs_combo/r4`): **0 erros**; sem o switch, 32.705.
- `route_mix_r4.py`: **0 deliberados fora da rota do recorde 4**.
- Mix dos blocos 0-12: 36 / 25 / 39, contra 36 / 25 / 40 do recorde.

Achados:
- **O proxy min-cut não serve para deltas.** Para o em9 ele dá −4 no `kc16rm` e −9 na posição 0; a ILP dá −1. No
  recorde 3 o erro foi para o outro lado: `fin_rc3b_m` 105 no min-cut contra 98 na ILP, delta −6 contra −10. Ou
  seja, o min-cut erra o delta para os dois lados. A regra "ILP + 7" vale para o nível absoluto, não para diferenças.
- **Pegadinha do estágio 12** (`make_n32a4_decode_plan.sh` replaneja o estágio 12 sozinho com
  `PLAN_ILP_EXIT_CAP=46`):
  - com a receita em9, a ILP pôs o SPRU em v_16 (CF 20) e saiu em 46/d2, efetivo 48 > 46, e a checagem do script
    abortou. Com `PLAN_ILP_EXIT_CAP=44` saiu igual (`run_r4f_em9_ilp*.log`). O teto do `ilp.py:802-821` não morde
    nesse caso;
  - usei o estágio 12 do min-cut, planejado na mesma entrada 34/d1 (v_17, CF 10, saída 44/d2, o pouso do
    recorde). Ele também tem 1 refresh, então a contagem não muda. Uma submissão precisaria resolver isso no
    planner.
- O em9 ainda melhora os raise drops dos SparseB: drop médio 2,78 → 3,38, o que deixa cada um mais barato. Custo não
  medido.

## 3. Nossa softmax ainda acrescenta algo sobre a centrada? — Quase nada

Tabela por camada: `subagent/r4/sm_table.py`. Níveis l1+Σcheb+l2: recorde 3 222, recorde 4 188; `cand_rc3` 204 já
com o desconto do smrel (210 no recorde 3 com o mesmo desconto).

| peça | sobre o kc16r | evidência |
|---|---|---|
| split l1/l2 + faixas recalibradas | **0** | Medido (`split_centred.py`): na centrada a faixa da rodada 0 é exata e independente dos dados, `[e^(−U/δ2)/16, 1,05]`, e o δ2 mínimo é imposto pelo τ = 4,6. A regra do recorde reproduz os `ks` da config. δ2 ×2 deixa as 10 camadas centradas +3 níveis mais fundas; δ2 /2 viola o τ. Nas camadas 8 e 11 o `cand_rc3` só difere pelo smrel. |
| smrel (exp minimax relativa, 1 quadratura a menos) | −12 níveis/token; ~−2 refreshes no proxy | Precisa de um ajuste: nas camadas centradas a série tem de valer 1 no topo (×e^(−A)), senão quebra o "exp ≤ 1" e a faixa da rodada 0 (`build_r4cfg.py`). KL float64 em 8 janelas OWT: **2,190e-6 contra 2,189e-6 do recorde 4** (`kl_r4.txt`). Min-cut na posição 0: em9 104 → 102; base 113 → 111 (com gelubs). Sem ILP. |
| gelubs | 0 a +1 no proxy | KL 3,59e-6 (×1,64). Não compensa. |
| defer_norm 1,3 | não medido; no recorde 3 ≈ −1 | O porte do d270f7a dá 2 conflitos textuais em `config.py`; o `attention.py` mescla sozinho. A centragem acontece no `complex_qkt`, antes da softmax, então é ortogonal ao adiamento. Não testado. |

O smrel e o defer_norm **não dependem do em9**: valem sem o switch, sobre o EvalMod do recorde.

## 4. Tempo contra o recorde 4 (estimativa)

Custos por rota: `plan/data/route_table_n32.json` (`bts_ms`) × razões do README do recorde 4 (0,84 / 0,82 / 0,81)
⇒ StC-first 14,31, SparseB 14,26, SPRU 8,08 ms. Esse modelo dá 1.430 ms de bootstrap/token para o recorde 4 e 1.885
ms para o recorde 3. A razão, 0,76, bate com o placar real (1,798/2,325 = 0,773). Script: `time_model.py`.

| em9 sozinho | ms/token |
|---|---|
| −1 SparseB-512 | **−14,3** (−0,8 % de 1.798) |
| EvalMod dense +20 % de limb-products (auditoria R3) × 42 refreshes dense × 14,31 ms × fração do EvalMod no bootstrap (10 / 20 / 30 %) | +12 / +24 / +36 |
| pousos 2 limbs mais ricos (ops fora do bootstrap) | + (não medido; a auditoria estimava ~+6 % por op) |
| raise drops maiores nos SparseB | − (não medido) |
| **saldo** | **−2 a +22 ms ⇒ −0,1 % a +1,2 %**: não chega a 1 % mais rápido |

smrel (+ defer 1,3) sem em9: se render −1 a −3 SparseB, dá −14 a −43 ms (0,8-2,4 %). Fica na margem, e é
estimativa pelo proxy, que já errou deltas.

## 5. Custo do porte (medido)

- 67717aa: cherry-pick limpo.
- d8a7643: 4 conflitos triviais, resolvidos em fc19e09:
  - contagens de campos nos testes: chain.preset + bts.evalmod ⇒ Config 260, envs 352;
  - `chain_of` e `fakecap_landings_of` lado a lado em `pc-py/src/shadow.rs`;
  - `BtsConfig::n32(em)` em `bts_all.rs`.
- Build ok; os 14 estágios compilam.
- Testes:
  - `pc-shadow --lib` 4/4 (`multi_stage_landings`, `evalmod_series_levels`);
  - `pc-config field_table` ok;
  - `shipped_stamps_round_trip` falha por `PLAN_PCMM_PREDIV_CAP` (`gpt2_prefill128_n32a4_pfbh_cm`). O
    `fields.rs` do próprio `origin/speedrun` também não tem esse campo, então a falha vem do upstream. Só inferido
    pelo grep, não rodei o teste no `origin/speedrun` puro.
- Não rodados: pytest, `prepare()` e a GPU.

## Caminho mais barato daqui

1. **Largar o em9 no Decode-8**: 1 refresh não paga o EvalMod dense mais caro. O commit fica pronto no
   `perseus-r4:r4-em9` se um recorde futuro voltar a ter SparseB sobrando.
2. Se quiser testar o resto, o smrel (com a normalização ×e^(−A) nas camadas centradas), talvez com o defer 1,3,
   **sem** o switch:
   - primeiro um par de ILPs na posição 0 (blocos 0-11; a captura cabe em 5G até o bloco 11, e os blocos 12/13
     vêm do recorde): base contra smrel;
   - só se der ≥ 2 refreshes, a captura fundida. Ela não cabe aqui: a da posição 0 já estourou 5G no bloco 12.
   - Ganho esperado ≤ ~2 %, e só a GPU decide.

## Artefatos

- `subagent/r4/`:
  - scripts `plan_count_r4.sh`, `plan_n32a4.sh`, `cap0.sh`, `route_mix_r4.py`, `time_model.py`, `build_r4cfg.py`,
    `sm_table.py`, `split_centred.py`, `emscan4/`;
  - logs `run_*.log`, `log_*.txt`, `p0_counts.txt`, `kl_r4.txt`, `emscan4.out`.
- `work/perseus-r4` (não rastreados):
  - planos `bootstrap_placements/{r4x_*,r4f_*,r3x_*,p0_*}`;
  - capturas `graphs/r4cap_*` (blocos 0-11; nas smrel, os blocos 12/13 são symlinks do recorde);
  - configs `configs/model/approximation/r4_{smrel,gelubs,smrel_gelubs}`;
  - 4 temporários `graphs/_b12_*` deixados pelo fluxo, sem `rm`.

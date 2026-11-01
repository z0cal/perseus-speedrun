# Teste de sanidade 3: EvalMod par, GELU par e softmax centrada (2026-11-01)

Revisão independente das alegações da sessão `claude/busy-hypatia-qp8dm8` (4 commits sobre 269d6c9). Rodou tudo na CPU,
sem GPU. Usei o upstream d97a6f7 (clone no scratchpad) e o branch `tmodel-gpu` 34e8db7, buscado do
`gpu/perseus-tmodel-entry.bundle` para o clone como `refs/remotes/sanity3/tmodel-gpu`. Não fiz commit, push nem edição
no repo; só escrevi este arquivo. Scripts auxiliares em
`/tmp/claude-0/-home-user-perseus-speedrun/36615c21-3150-568c-8729-63c307d6d65d/scratchpad/sanity3/` (abaixo `$S`).

## Veredito geral: **GO condicional** para implementar `EvalMod::Even` e levar à GPU

- A matemática do pré-passo está certa.
- A convenção da entrada (u/K em [−1, 1], c0 da tabela não dividido por 2) bate com o código real.
- As listas de ops conferem.
- O modelo de tempo reproduz −37,8 / −22,6 / −60,3 ms. O mix de sítios bate com o código do compilador e com os grafos.

Nada disso mede tempo real: o ganho continua sendo de modelo. Antes do A/B, meça na GPU `pc-bench bts`, com ms por
etapa e `bts-prec` nos modos real e complexo. Ressalvas:

- **(a) Precisão:** o par z11 dá 33,7–34,2 bits, não 34–35.
- **(b) Ganho do drop:** ~25 % do ganho E (−9,4 de −37,8 ms) vem do raise e da CtS 2 limbs mais baixos, ou seja, do
  drop. Esse efeito só foi validado no SparseB.
- **(c) Implementação:** fica sobre o `tmodel-gpu`, que tem `EvalMod`/`shift_evalmod`. O d97a6f7 não tem nenhum dos
  dois. Faltam:
  - `levels() = 9`, para o Shadow re-pousar;
  - `FIDESLIB_BTS_RAISE_DROP=0:1,512:2`;
  - +1 nos `raise_drop` densos e 512 dos placements;
  - o 6 em `FIDESLIB_BTS_RAISE_DROPS`.
- **(d) Achado novo:** `FIDESLIB_BTS_SPARSE_B_K=16` já existe e o recorde de Prefill 3 já o usa. No mesmo modelo, ele
  dá −31,2 ms no SparseB sem nenhum código novo. Com o par z11 K16 + 1 drop também no SparseB, o total chega a
  −98,2 ms. Vale testar na mesma sessão de GPU.

| # | alegação | veredito |
|---|---|---|
| 1 | Pré-passo par correto; entrada u/K; c0; bits (z11 K16 34–35; z13 K24 e6 30–33; recorde 27,4 / 35,2); eps 2^-6 | **CONFIRMADA**, com bits do z11 um pouco otimistas (33,7–34,2) |
| 2 | Listas Shipped/D27 idênticas às do `tmodel/em`; z11 = 18 limbs, 10 relins, 15 rescales (recorde 20, 11, 18) | **CONFIRMADA** (na convenção do censo; ver nota de nl) |
| 3 | −37,8 / −22,6 / −60,3 ms; cenário A −9,4 = relatório; sítios, drops e pousos | **CONFIRMADA** como modelo; 2 desvios de nível no `tokplan` (conservadores), requisitos de implementação não ditos |
| 4 | GELU 210 → 180 (179); `IMPL_DEAD_OPS` pula os ímpares; ×2 de z grátis | **CONFIRMADA**; 179 perde 1 nível na camada 7 (180 é o "mesmo nível") |
| 5 | Softmax centrada: ref na linha (u = 0); erro absoluto em u ≤ 0; sem quadratura não alcança; 1 a menos, grau 10–13 | **PARCIAL**: fatos e script conferem e a simulação é exata na forma, mas o erro real do recorde é 35–185× maior (Goldschmidt) |
| 6 | Limpeza sem dependências; documentos consistentes | **PARCIAL**: nenhum código depende dos removidos; 3 inconsistências pequenas nos `.md` |

---

## 1. Pré-passo par do EvalMod: CONFIRMADA

**Matemática.**
- `tmodel/emcost_even.rs` (idêntico a `<clone>/crates/pc-shadow/examples/emcost_even.rs`, `diff` vazio) faz, com
  x = u/K na entrada:
  1. `add_scalar(x, −0,25/K)`, que dá (u − ¼)/K;
  2. `square`, `add(x, x)` e `add_scalar(−1)`, que dão z = 2((u − ¼)/K)² − 1.
- Isso é exatamente o `em_even.pipeline_even` com L = K.
- A união de intervalos fica com |u − ¼| ≤ K − ¾ + eps < K, então z ∈ [−1, 1).
- A fusão (`EmProgram::fused`) junta `Mult` + `Add(d, d, d)` + `AddConst(d)` no epílogo da relinearização. O ×2 e o
  −1 saem grátis.

**Convenção do código real:**
- **Rota densa:** `crates/pc-gpu/src/bts/dense.rs:121` usa
  `exact = sf(raise_top)/qd · 1/(boot_k·n)`, então o EvalMod recebe u/K.
- **SparseB:** `sparse_b.rs:273` usa `gc = raise_k/evalmod_k · 2^-t`, sobre a constante do raise feita com
  `raise_k`. A entrada fica u/24.
- **Tabelas enviadas:** a docstring de `EVALMOD_CHEB` diz que a tabela é o ajuste de
  `(2π)^(−1/32) cos(2π(16y − 0,25)/32)`, com y = u/16.
- **c0:** `pc_core::cheb` declara "OpenFHE's (c0 NOT halved in storage; the evaluator adds c0 / 2)". O
  `build_chebyshev_m2` soma `divcs_q[0]/2` e `divqr_q[0]/2`, e o `long_division` faz `q[0] *= 2`. Os fits `even_*.json`
  saem de `fit_even` com `T[:,0] *= 0,5`, ou seja, na mesma convenção.

**Avaliação numérica independente.**
- `$S/emchain.py` porta para Python o `long_division` e o `PsPlan` do `pc_core::cheb`. Porta também a semântica de
  valores de `build_chebyshev_m2`, `build_chebyshev_general` e `inner_ps`, incluindo o quirk do `wsum` de s2. Os ângulos
  duplos usam os escalares de `build_double_angle`.
- Rodei `$S/run_prec.py` com o venv. A tabela dá bits = −log2 max|saída − sin(2πu)/2π| em u = I + δ, |I| ≤ K − 1,
  |δ| ≤ eps:

```
cfg                        k,m #mul  2^-1  2^-2  2^-3  2^-4  2^-5  2^-6  2^-7  2^-8 2^-10 2^-12
ship K16 d14 r5          (5, 2)   12  25.4  27.4  27.4  27.4  27.4  27.4  27.4  27.5  27.5  27.5
d27 K16 r4               (4, 3)   14  43.1  43.0  43.3  43.2  43.4  43.4  43.4  43.4  43.4  43.5
even z11 K16 r4          (4, 2)   11  33.7  33.7  33.7  33.9  34.0  34.1  34.1  34.1  34.1  34.2
even z13 K16 r4          (2, 3)   13  42.3  43.2  43.5  43.7  44.0  43.8  43.8  44.0  43.8  43.9
ship K24 d20 r5          (3, 3)   14  34.4  34.4  34.6  34.9  35.0  35.1  35.2  35.2  35.2  35.2
d27 K24 r4               (4, 3)   14  28.2  28.2  28.5  28.8  28.9  29.0  29.1  29.1  29.1  29.1
even z13 K24 r4 e6       (2, 3)   13  23.9  29.7  30.1  31.5  32.3  32.7  32.9  32.9  33.0  33.0
```

O porte do PS reproduz o `em_even.err_even` (33,7–34,2 para o z11). Valores na tabela:

| série | medido aqui | alegado |
|---|---|---|
| recorde K16 | 27,4 bits | 27,4 ✓ |
| recorde K24 | 34,4–35,2 bits | 35,2 ✓ |
| par z13 K24 e6, \|δ\| ≤ 2^-3 | 30–33 bits | 30–33 ✓ |
| par z13 K24 e6, 2^-2 / 2^-1 | 29,7 / 23,9 bits | — |
| **par z11 K16** | **33,7–34,2 bits** | 34–35 (o README diz 34,1–35,3); otimista em ~0,5–1 bit |

**Ruído.**
- `$S/run_noise.py` injeta ruído gaussiano σ após cada produto ct×ct e na entrada.
- Com σ = 2^-45, a amplificação rms/σ dos produtos é ~110 para shipped, d27 e z11 (K16), e ~170–200 para shipped,
  d27 e z13 (K24).
- O pré-passo não piora o condicionamento.

**eps 2^-6 do fit K24.**
- O `plan/btserr.py` do upstream põe a mensagem na banda [0,003, 0,03]·2^CF, então |δ| ≲ 2^-5. Nessa faixa o
  z13 K24 e6 dá ≥ 32,3 bits. É compatível.
- Na cauda |I| = K, o z13 K24 e6 cai para 2^-20,6 em I = −24, contra 2^-38,5 do recorde (`$S/edge.py`). O comentário
  de `sparse_b.rs` diz que o inteiro após o fold tem a distribuição da rota densa (|I| < 16), então isso não deveria
  ocorrer. Mesmo assim, é um ponto para o `bts-prec`.
- O z11 K16 em I = ±16 fica em 2^-27 / 2^-34,6, melhor que o recorde (2^-24,7 / 2^-25,0).

## 2. Listas de ops: CONFIRMADA

`$S/cmp.py` regera todas as listas com o binário do upstream:
- comando:
  `CHAIN=n32a4 TOP=t emcost_even [D27_K16 full_d27_r4 4 0 D27_K24 k24_full_d27_r4 4 0 Even_z11_K16 … 4 16 Even_z13_K16 … 4 16 Even_z13_K24 even_k24_z13_r4_e6 4 24]`;
- comparação: `oplist`, `relin` e `levels` contra `tmodel/em/n32a4_t*.jsonl` e `even_n32a4_t*.jsonl`;
- resultado: **todas OK em t = 30…48**;
- topo 42: z11 = 18 limbs, 10 relins (334), 15 rescales; recorde = 20, 11 (359), 18.

**PS e fusão:**
- O z11 usa k = 4, m = 2, na forma de nível único (a mesma do recorde, k = 5, m = 2, com teste bit a bit contra o d3).
- O z13 usa k = 2, m = 3, no caminho geral. É o mesmo caminho do K24 enviado (k = 3, m = 3), que já roda no recorde.
- O censo usa `fused()`. A GPU usa `rewrite(fused())`, que ainda tira cópias, e `PC_EM_FUSE_RESCALE=1`. Os dois tratam
  todas as séries igualmente, então nada escondido inverte o sinal.

**Nota de convenção:**
- O censo usa entrada nl = 2. Na rota densa, porém, a entrada real é nl = 1: `dense.rs` reescala após cada estágio da
  CtS, e o log do `bts_prec` mostra "em in top 45, nl 1".
- Com nl = 1 (`$S/nl1_*`, exemplo `sanity3_emcost_nl` no clone):
  - recorde @42: 11 relins (381), 16 rescales;
  - z11 @40: 10 relins (334), 14 rescales.
- Os "15 vs 18" incluem 1–2 rescales de entrada a 43 limbs que não existem no runtime.

## 3. Estimativa de tempo: CONFIRMADA como modelo, com 2 desvios e requisitos não ditos

`cd tmodel && python3 -I even_sites.py <clone>/bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm` reproduz:

- sítios: `{'512': 40, '1': 26, 'stcfr': 36, 'stcf': 6, '1r': 6}`, total 114;
- A: −9,4; A': −22,6; densas d27: +21,0; **E: −37,8** (stcfr −29,4, stcf −8,4); z13 densas: +6,4; **E + A': −60,3**
  (−3,57 %);
- −9,4 é o valor da opção A em `tmodel/RELATORIO-2026-10-30.md` (tabela da seção 4, `sbd27d1_cm`).

**(a) Reconstrução dos sítios.** Conferida no código e nos grafos do upstream.
- **Deliberados dos blocos 0–11:** o nó `deliberate_bootstrap` da `kv_push`
  (`graphs/gpt2_decode_python_n32_kc16rm/block_*/graph.json`) tem `"args": {"route": {"slots": 0, "stc_first": true, "real": true}}`.
  - O compilador (`pc-program/src/compile/apply_plan.rs:282-291`) marca `route.real = true` com `FIDESLIB_BTS_REAL=1`.
  - A rota sai de `route_for_landing` com `base(0)`, StC-first com drop padrão 0.
- **Os 6 deliberados do bloco 13** têm `slots: 1`, pouso capturado 14 e vão para a SPRU.
- **Os 6 `hint_fire` do bloco 13** não estão em `real_route` (o bloco nem tem a chave), então são complexos.
- **Drops dos placements:** são absolutos (`route_tagged`: `r.raise_drop = d`). A base vem de
  `FIDESLIB_BTS_RAISE_DROP` por rota (`RouteEnv::base`), 512:1 no recorde e 0 nas densas.

**(b) Desvios de nível no `tokplan.site_units`.** Ambos são conservadores.
- **SparseB:** o EvalMod é precificado em 48 − 2d com lista nl = 2. O runtime usa
  `em_in = FMeta{top: raise_top = 50 − 2d, nl: 2}` (`sparse_b.rs::tops`), ou seja, 1 nível acima.
- **Densas:** o modelo usa a lista nl = 2 em 42 − 2d; o real é nl = 1 em 42 − 2d.
- Efeito nos deltas por cadeia (`$S/em_ms.py`):
  - densas: −0,589 (modelo) contra −0,592 ms (real), sem efeito;
  - SparseB z13: −0,470 contra −0,582 ms, então o A' seria ~−27 ms em vez de −22,6.
- Real = 1 cadeia e complexo = 2 faz sentido. O pc-bench (StC-first, real = false, K = 1) dá EvalMod 8,20 ms com as
  2 cadeias juntas, ~4,1 ms por cadeia. O `calib.py` calibra exatamente `em_shipped_x2` = 8,20.

**(c) Pouso.** Com os metadados reais:
- denso: recorde @42 nl1 → out top 24 nl2; z11 @40 nl1 → 24 nl2 (igual);
- SparseB: recorde @48 → 28; z13 @46 → 28 (igual).

**(d) Drops.**
- A receita termina em `PLAN_RAISE_ROUTES=0,512,1`. Há sítios densos com drop: v_818 e v_816 (bloco 3) com 1, v_955
  com 3 e v_54 (complexo, bloco 13) com 2.
- Com +1, as densas vão até 4, dentro de `FIDESLIB_BTS_RAISE_DROPS=1,2,3,4,5`. O SparseB vai até 6 e precisa do 6,
  como faz o `sbd27.py`.
- O número de variantes não cresce: {0, 1, 2, 3} vira {1, 2, 3, 4}.
- O `IDEIA-evalmod-par.md` não diz, mas a implementação também precisa disto:
  1. `EvalMod::Even` com `levels() = 9` no `shift_evalmod`/`evalmod_landing_shift`, que só existe no branch
     `tmodel-gpu` (d97a6f7 não tem);
  2. `FIDESLIB_BTS_RAISE_DROP=0:1,512:2`. Sem isso, os 12 deliberados K/V/token (pouso capturado 26) não acham rota
     densa pelo `route_for_landing`;
  3. +1 nos `raise_drop` densos e 512 dos placements, nunca nos SPRU (v_89: 12, v_674: 6 etc.);
  4. o flag do pré-passo, com o K da rota, em todos os `build_evalmod`/`EvalModPlan::program`/`table_bytes`
     (`dense.rs`, `sparse_b.rs::tops`/`em_bytes`/`new`, `sparse_classic.rs`, `pc-tools`).

**(e) O que o modelo não vê.**
- **Decomposição do E** (`$S/decomp.py`; real, drop 0: −0,813 ms/refresh):
  - −0,589 ms é do EvalMod (−12 % pela série, mais −4 % pelo topo 2 limbs abaixo);
  - −0,224 ms é do raise e da CtS 2 limbs abaixo, ou seja, **−9,4 ms/token só do drop**.
  - O efeito do drop só foi validado no SparseB (`sbd27d1`, ×1,007).
- **Calibração:** um só ponto por série (d27). O proxy não separa o custo de Mult, Rescale e WSum.
- **Memória:** os buffers e tabelas do EvalMod ficam fora do modelo. O d27 subiu `bts.work` em 1,5 GB no n32.
- **Variantes novas:** o drop 4 na StC-first nunca rodou no recorde.
- **Precisão no `bts-prec`:** o "complex |out − in|" do bench dependeu da tabela (shipped 2,82 bits contra d27
  6,34 bits em A = 0,01). Meça o par nos modos real e complexo.
- **O quadrado extra já está na lista de ops**, e o ruído foi conferido no item 1.

**Achado lateral (não alegado).**
- `FIDESLIB_BTS_SPARSE_B_K=16` existe em d97a6f7. O Prefill record 3 (`BabyStepPrefill`) já o usa, declarado como
  "approximation change", e `sparse_b.rs` diz que o inteiro após o fold tem a distribuição densa.
- Com `$S/even_sites_x.py`, mesmo modelo:
  - tabela K16 enviada no SparseB, sem drop e sem código: **−31,2 ms**;
  - par z11 K16 + 1 drop no SparseB: −60,4 ms;
  - densas z11 + SparseB z11 K16, +1 drop: **−98,2 ms (−5,8 %)**, com uma tabela só.
- Precisa de `bts-prec` e KL. O K16 no SparseB fica com 27,4 bits, ainda abaixo do piso do SparseB.

## 4. GELU: CONFIRMADA

- `python3 -I scripts/gelu_bsgs_count.py <cfg r4>`: atual 210, par em z no mesmo nível 180. A tabela por camada é
  igual à do `ATAP-ganhos.md`.
- `PYTHONPATH=<clone>/python venv tmodel/gelu_count.py <cfg>`, com o `poly.py` real: 210 / 179.
- `subagent/depth_trace.py` adaptado (`$S/depth_trace_r4.py`, `poly.py` do clone, config kc16r_cm): 210 em x.
  Os graus aparados (62/94/254/60) e as profundidades (7/7/9/6) conferem.
- **179 contra 180:** com o m padrão, a camada 7 (grau 60) sai a 7 níveis em z contra 6 hoje. O 179 perde 1 nível; o
  180 (m escolhido, (6, 13)) é o número "sem perder nível".
- **`IMPL_DEAD_OPS`:** padrão 1 (`impl/ops.py:136`) e ligado na receita. Os ímpares ≥ 3 valem ≤ 5,5e-16, contra o corte
  `cheb_eps·max|c|` = 5e-10, e viram 0 no `_cheb_trim`. O `_cheb_needed` então pula os babies ímpares.
- **×2 de z:** z = T_2(x) é `add(add(sq, sq), −1)` no próprio `poly.py`, sem nível.
- `GELU_FOLD=1` só dobra 1/xmax na projeção up. A série roda em [−1, 1].

## 5. Softmax: PARCIAL

- **Camadas centradas:** 0–7, 9 e 10, em que `ref_keys` não está vazio na config. As camadas 8 e 11 não têm `ref_keys`.
- **A referência está sempre na linha:**
  - `ref_key(kind, kc)` dá 0 (sink), kc − 1 (self) ou max(kc − 2, 0) (prev), todos em `range(kc)`, que é exatamente o
    que `active_mask` liga;
  - `_ref_check` recusa kc > `ref_window` = 16;
  - os slots de transbordo são zerados pelo `active_mask` após o exp.
- Logo Σ e^{u_j} ≥ e^0 = 1 e p_i ≤ e^{u_i}, a menos do erro do próprio exp em u = 0. O argumento do critério misto
  procede.
- **`sm_seed_search.py`** (venv, ~10 min) reproduz a tabela da nota:
  - recorde, max|Δp| simulado: camada 0 2,3e-5, camada 2 9,9e-6, camada 9 6,0e-6, camada 5 2,3e-6;
  - k1 = 0: nenhum d ≤ 15 alcança;
  - k1 − 1: d = 11, 10, 11 e 13, ou seja, grau 10–13.
- **A cadeia simulada é exata na forma.** `$S/sm_cmp.py` compara `sm_seed_sim.chain` com `examples/gpt2/ref.py::softmax_ref`
  do upstream, a emulação em claro do `softmax_thor`, com `SM_CHEB_GS=1` e `SM_GS_FIRST=1` como no recorde:
  - depois de renormalizar as linhas, max|p_ref − p_sim| = 2e-15 nas camadas 0 e 5;
  - **porém** o `softmax_ref` cru tem max|p − p_exato| = **8,2e-4 (camada 0) e 4,3e-4 (camada 5)**, com Σp ∈ [0,99918, 1,00082].
    É o erro relativo minimax do Goldschmidt escalado, um fator por linha, 35–185× acima do erro de forma que a busca
    usa como alvo.
- **Consequência:**
  - a conclusão "sem quadratura não alcança o recorde" vale para o critério escolhido, o erro de forma;
  - "linha fechada" é mais forte do que o teste mostra: o critério é muito mais estrito que o erro real do recorde;
  - o README chama o alvo de "erro real do recorde", o que é impreciso;
  - se um erro de forma ~1e-4 é aceitável no KL, só o gêmeo ou a GPU dizem.

## 6. Limpeza: PARCIAL

- **Dependências:**
  - `grep` por `HANDOFF.md|fin_dfr13_m_cm|gpt2_decode_em8|headroom_split|sm_split_v1` em `.py`/`.sh` não acha nenhum
    import nem caminho;
  - as únicas menções restantes estão em docstrings (`band_tail.py`, `sm_split.py`, já marcadas como "removed") e em
    `.md` históricos (auditoria, `sanity/`, `r4/`);
  - `final_plan.sh`, `plan_count.sh` e `plan_count2.sh` usam `plans/gpt2_decode_em9{,_cm}`, que continuam no repo.
- **Números consistentes:** −37,8, −22,6, −60,3, −3,6 %, −9,4, 210 → 180, −30 produtos, 30–33 bits e 10 relins / 334 /
  15 conferem entre README, ATAP, IDEIA-* e HANDOFF-2026-10-30.
- **Inconsistências:**
  1. **Bits do par z11:** 34–35 (IDEIA/ATAP) e 34,1–35,3 (README, 10-30 noite), contra 33,7–34,2 com o
     `fits/even_k16_z11_r4.json` (`em_even.py` dá o mesmo na própria linha).
  2. **GELU:** o README diz que o `gelu_count.py` "agora bate" com o `gelu_bsgs_count.py`. Bate em x (210), mas em z dá
     179 contra 180, pela diferença de nível na camada 7.
  3. **Softmax:** "erro real do recorde" (README, item da softmax de 2026-11-01) é o erro simulado de forma. Ver item 5.

## Bugs e riscos (resumo)

**Bugs:**
- **Nível do `tokplan.site_units`:** EvalMod do SparseB 1 nível abaixo do runtime; densas com lista nl = 2 em vez de
  nl = 1. Conservador nos deltas: A' ~−27 em vez de −22,6.
- **Documentos:** bits do z11 superestimados em ~0,5–1 bit; 179 contra 180 na GELU; "erro real" da softmax.

**Riscos para o GO:**
1. **Ganho só de modelo:** ~25 % dele é efeito de drop nas rotas densas, nunca medido.
2. **Implementação no Shadow:** sem `levels() = 9` e o drop base 0:1, os deliberados não compilam com pouso igual.
3. **Variantes novas:** drop 4 denso e drop 6 SparseB.
4. **Precisão:** `bts-prec` (real e complexo) e KL não medidos; cauda |I| = 24 do z13 K24.
5. **Memória** dos buffers do EvalMod.

**Ordem sugerida na GPU:**
1. `pc-bench bts` (ms + `bts-prec`) para:
   - o par z11 +1 drop;
   - o z13 K24 +1 drop;
   - **`FIDESLIB_BTS_SPARSE_B_K=16`**, que custa zero código.
2. Depois, o A/B do pacote contra o recorde 4.

## Arquivos desta revisão

`$S` = `/tmp/claude-0/-home-user-perseus-speedrun/36615c21-3150-568c-8729-63c307d6d65d/scratchpad/sanity3`

| arquivo | conteúdo |
|---|---|
| `$S/emchain.py` | emulador do EvalMod: PS portado do `pc_core::cheb` e do `pc_shadow::evalmod`, com injeção de ruído |
| `$S/run_prec.py`, `$S/run_noise.py`, `$S/edge.py` | precisão por eps, ruído, inteiros de borda |
| `$S/cmp.py`, `$S/em_ms.py`, `$S/decomp.py`, `$S/even_sites_x.py` | listas, convenção nl/topo, decomposição, cenários extras |
| `$S/sm_cmp.py`, `$S/sm_seed_search.out`, `$S/depth_trace_r4.py` | softmax contra `softmax_ref`, saída da busca, GELU com o `poly.py` real |
| `<clone>/crates/pc-shadow/examples/sanity3_emcost_nl.rs` | `emcost_even` com `NL` selecionável (no clone do scratchpad) |

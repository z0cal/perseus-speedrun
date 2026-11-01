# Softmax do recorde 4: segunda opinião (subagente, 2026-11-01)

Tudo rodou na CPU com o código do upstream (`d97a6f7`). Usei `examples/gpt2/ref.py::softmax_ref` com `SM_CHEB_GS=1` e
`SM_GS_FIRST=1`, o gêmeo float64 da `softmax_thor`. Também usei os grafos capturados e os placements do recorde 4, que
estão no clone (`graphs/gpt2_decode_python_n32_kc16rm`, `bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm`).

As linhas de scores são sintéticas, com kc ∈ {2, 4, 8, 16} e seis tipos: uniforme na faixa, agrupada, "real" (N(μ, σ)
com μ ∈ [−8, 2]), plana perto da referência, topo e fundo. Para as camadas centradas, a referência fica em u = 0. Cada
métrica é o máximo sobre os tipos e os kc. O V é aleatório, N(0, 1) com 64 colunas, e dout = ‖Δ(p·V)‖/√64.

Não houve captura, ILP, KL do modelo nem GPU. Scripts e logs estão em `subagent/softmax2/scripts/`. O harness
(`sm2.py`) bate com o `softmax_ref` byte a byte, e a réplica com injeção de ruído (`noise.py`) dá diferença 0.

## 0. Resumo e ranking

| # | ideia | onde cai | níveis / refreshes | ms (tmodel) | risco | próximo passo |
|---|---|---|---|---|---|---|
| 1 | **δ2 8 → 4 na camada 4**, com margem 1,0–1,1 (H3) | rodadas de refino da camada 4 | −1 SparseB, −4 a −5 níveis | **≈ −8,4 ms (−0,5 %)** | ruído do SB#1 da camada 4 ×2,8 a ×4,5 (rms); tolerância acima de U\* cai de 1,57 para 1,16–1,26 U\* (acima disso, NaN) | na sua máquina: cauda de max u da camada 4 no hold-out, KL do modelo de ruído e ILP |
| 2 | **normalização adiada (defer_norm) sobre o recorde 4** (fora das hipóteses) | trecho rodada 1 → softmax_v → out_proj → LN2 → refresh denso | −2 níveis por camada exatamente no trecho que tem de encolher para o SB#3 sumir | 0 a −8,4 ms por bloco em que o SB#3 suma; teto ~−90 ms | CutMax quebrou com defer na camada 11 (recorde 3); no recorde 4 nunca foi para a ILP | portar `defer-norm` para o r4 e rodar a ILP por camada |
| 3 | **Goldschmidt re-dividido** com a mesma profundidade (H2 revisto) | rodada 0 (−1) / rodada 1 (+1) | 0 no total | 0 sozinho | nenhum para a precisão (o fator por linha cai 100–170×); na ILP gasta folga da rodada 1, que é justamente o trecho do item 2 | só config; KL no gêmeo |
| 4 | **semente do exp pelo critério misto** (H1) | trecho do q (LN → qkv → qkᵀ → exp → SB#1) | −2 níveis por camada em 5–7 camadas, −1 nas outras: ≈ −17 a −19 níveis/token (smrel: −12) | **≈ 0** com o drop search atual (precedente do smrel); até ~−10 ms se a folga virar drop | robustez abaixo de clip_lo pior que a do recorde; com k1 = 0, só a camada 4 passa | entra no pacote de folga, com o item 2 |
| 5 | margem superior 1,25 → 1,1 nas camadas centradas | rodada 0 | −1 nível nas camadas 0, 2, 4, 6 e 9 (−5/token) | ≈ 0 | tolerância acima de U\* cai de ~1,45 para ~1,3 U\* | só com a cauda dos dados |
| 6 | afrouxar os alvos do Goldschmidt (H2 original) | — | os alvos intermediários não mexem na precisão; o final a 1e-2 tira −1 em 4 camadas | ≈ 0 | erro ×12 | dominado pelo item 3 |

Nada disso, sozinho, chega aos 1 % da regra 5. O item 1 é o único que tira um refresh por conta própria. O item 2 é o
único que ataca o trecho em que, pelas contas de nível abaixo, um SparseB por bloco pode sumir.

## 1. Fatos conferidos (e correções)

**Erro real do recorde por camada** (métricas de `sm2.evaluate`). O erro total é dominado pelo fator por linha da
última rodada do Goldschmidt, e esse fator **varia 15× entre as camadas**:

| camadas | contagens GS | fator \|Σp − 1\| | max\|Δp\| | dout | erro de forma (do exp) |
|---|---|---|---|---|---|
| 0, 2, 6, 9 | [7, 4] | 8,2e-4 | 8,2e-4 | 9,8e-4 | 2,4e-5 / 1,1e-5 / 4,4e-6 / 6,8e-6 |
| 4, 5, 10 | [6,2,3,5], [6,5], [6,5] | 4,2e-4 | 4,2e-4 | 5,1e-4 | 1,1e-5 / 2,4e-6 / 6,5e-6 |
| 1, 3, 7 | [5,5], [6,5], [6,5] | **5,5e-5** | 5,5e-5 | 6,5e-5 | 5e-7 / **1,5e-5** / 9e-7 |

Os seus números (8,2e-4 na camada 0, 4,3e-4 na 5) conferem. Só que "+10–50 % do erro do recorde" é um orçamento 15×
mais apertado nas camadas 1, 3 e 7. Na camada 3, a semente atual já consome 27 % do erro total.

**Contagem de produtos errada no BSGS.** `scripts/sm_seed_weighted.cost` e as tabelas da nota usam
`bsgs([1.0]*(d+1))`. Com coeficientes todos iguais a 1, o `split` cancela o resto exatamente e o `IMPL_DEAD_OPS` poda
produtos que não somem com coeficientes reais. Com coeficientes genéricos (ou os do LP), a contagem fica:

| grau | 5–6 | 7 | 8 | 9–11 | 12–14 | 15 |
|---|---|---|---|---|---|---|
| (níveis, ct×ct) reais | (3, 4) | (4, 4) | (4, 6) | (4, 6) | **(4, 7)** | (5, 7) |
| com `[1.0]*(d+1)` | (3, 3) | (4, 4) | (4, 6) | (4, 5) | (4, 5–6) | (5, 6) |

Consequências:
- (14, 0) custa **(4, 7)**, não (4, 5);
- o smrel (14, 1) custa (5, **8**), o mesmo número de produtos do recorde (6, 8). Ele tira 1 nível e 0 produtos, não 2;
- (6, 3) custa (6, 7) e (9, 2) custa (6, 8): os dois "−1 / −2 produtos" da nota não existem.

O grau 7 (4, 4) com o mesmo k1 tiraria 2 produtos do grau 8. É pouco.

**Estrutura de níveis por bloco** (`seg_slack.py`, `trace_sm.py`). Cada camada com δ2 = 2 tem 3 SparseB:
- SB#1 na saída do exp (o `square`);
- SB#2 na saída da rodada 0 (y = z·r);
- SB#3 na saída da rodada 1 (o p final).

Pousos, em limbs: SB#1 em 26–28 (folga 0–1 nível no trecho da rodada 0), SB#2 em 30–34 (folga 2–3 no trecho da
rodada 1), SB#3 em 30–34. As camadas 4 e 11 têm 5 SparseB, por causa das duas rodadas de refino a mais. São 40 SparseB
por token, o que confere com o tmodel. Os trechos têm ~10 níveis (pouso 26 → entrada ≤ 46–48).

A rodada 0 ocupa k0 + 2 = 8–10 níveis e a rodada 1 ocupa k1 + 3 = 7–8. Os três trechos são rígidos:
- exp + rodada 0 não cabem num trecho;
- as rodadas 0 + 1 (≥ 15 níveis) também não;
- o único refresh da softmax que pode sumir é o **SB#3**, se o trecho fundido [rodada 1 (7) + softmax_v → out_proj →
  LN2 vetorial → refresh denso do `up` (6, medido no bloco 0: 32 → 44)] = 13 níveis cair para ≤ 11.

Isso pede −2 a −3 níveis **nesse** trecho. A semente do exp (H1) está no trecho do q, então não ajuda aqui. É uma
hipótese minha, porque não rodei a ILP. Confira nos placements `slack_k2_sb_dense` qual dos três SparseB sumiu em
cada bloco.

## 2. H1: semente do exp contra o erro real (`seed_search.py`, `pacote.py`, `ext.py`)

Ajustei a semente por LP com bandas no critério misto (absoluto para u ≤ 0, relativo para u > 0, em unidades da
referência), com bissecção em ε, e medi com o `softmax_ref` de verdade. A tabela dá a razão (erro total do candidato) /
(erro total do recorde), no formato maxdp / dout / tv:

| camada (k1 do recorde) | (14, 0): 4 níveis, 7 prod | (14, 1): 5 níveis, 8 prod | (6, 2): 5 níveis, 6 prod |
|---|---|---|---|
| 0 (2) | 1,42 / 1,21 / 1,90 | 1,00 | 1,20 / 1,20 / 1,42 |
| 1 (2) | **1,01** / 1,00 / 1,02 | 1,00 | 1,37 / 1,39 / 1,75 |
| 2 (2) | 1,11 / 1,01 / 1,22 | 1,00 | 1,16 / 1,15 / 1,32 |
| 3 (2) | 3,8 / 3,2 / 7,2 (forma 2,2e-4) | 0,99 | 5,2 / 4,9 / 9,2 |
| 4 (2, δ2 = 8) | **1,00** (forma 9e-7) | 1,00 | 1,00 |
| 9 (2) | 1,06 / 1,00 / 1,13 | 1,00 | 1,14 / 1,13 / 1,29 |
| 10 (2) | 1,21 / 1,09 / 1,45 | 1,00 | 1,40 / 1,29 / 1,82 |
| 5, 6, 7 (3) | 24 a 136 (inviável) | **1,00** (−2 níveis) | 7 a 35 |

As sementes (14, 1) nas camadas 5–7 tiram **duas** quadraturas sem custo de precisão: −2 níveis e −1 produto por
camada.

**Combinado com o item 3** (Goldschmidt [6, 5] em vez de [7, 4], mesma profundidade), o (14, 0) passa com folga nas
camadas 0, 2 e 9. Na camada 0 o maxdp fica em 0,57× o do recorde, o dout em 0,48× e o tv em 1,14×; nas camadas 2 e 9,
abaixo de 0,25×. Na camada 10, com [6, 6] (+1 nível na rodada 1, que tem folga), fica em 0,29×. As bandas continuam
respeitadas em todos os pacotes:
- rodada 0: min/piso ≥ 1,00 e max/topo ≤ 0,83;
- rodada 1: min/piso ≥ 1,01 e max/topo ≤ 0,94.

A semente com o fator 1 no topo mantém z ≤ 1. O ruído do SB#1 não muda, porque depende de δ2 e U, não da semente.

**Fora da faixa: aqui o (14, 0) cai.**
- **Acima de clip_hi:** igual ao recorde. Quem manda é o topo 1,05 do Goldschmidt, que estoura logo acima: D = 1,2 já
  dá 1e17. O estouro acontece em U + ~0,74·δ2.
- **Abaixo de clip_lo** (a margem do ajuste é só 5 % da faixa abaixo do mínimo dos dados), em float64:

  | | recorde | (14, 0) sem guarda | (14, 0) com guarda de ~W/8 abaixo de lo |
  |---|---|---|---|
  | robustez | ≤ 8e-4 até lo − 10 em todas as camadas centradas (até −20 na maioria) | erro 0,6–0,9 já em lo − 2 (camadas 0, 2, 3, 9, 10) | robusto até lo − 5 |
  | erro de forma | — | — | 1,9e-4 (camada 9), 3e-4 (2), 4,2e-4 (10), 6,6e-4 (3), 1,4e-3 (0) |

  Com a guarda, a vantagem quase some, e na camada 1 nem a guarda resolve. **A camada 4 é a exceção:** com guarda de
  31 (W/4), o (14, 0) tem forma 2,3e-6 e fica robusto até lo − 20.
- **(14, 1) nas camadas 5–7:** com guarda de ~W/4, robusto até lo − 10 sem perder precisão (forma ≤ 2,6e-6).
- **(14, 1) nas camadas 0, 1, 2, 9 e 10:** robusto até lo − 5, mas erra 0,03–0,8 em lo − 10, mesmo com guarda. A causa
  é o `CHEB_COEFF_EPS = 1e-6` do `eval_chebyshev`. Ele zera coeficientes < 1e-6·max, o que dentro de [−1, 1] custa
  ≤ 1e-6, mas fora dele o termo cortado vira c·T_14(t): em t = −1,65, T_14 ≈ 2e6. Conferi na camada 9: o g(t) do LP é
  1,6e-4 em lo − 10, mas o p da chave sai 0,06. O ajuste tem de ser feito com o corte em mente (zerar e reajustar).
- No CKKS, qualquer entrada bem fora de [−1, 1] também é um problema de magnitude: os T_k intermediários passam de
  cosh(k·acosh|t|) e furam o max_abs capturado. A robustez do recorde em lo − 10 só vale no float. O que importa é
  quanto os dados reais passam de clip_lo, e isso só a sua máquina responde: o mínimo de u no hold-out contra o lo de
  cada camada.

**Conclusão de H1.** Pelo critério honesto (erro total ≤ recorde), sem quadratura passa só a camada 4, com folga e
robusto, e as camadas 1, 2 e 9 com ressalvas fora da faixa. Com uma ou duas quadraturas a menos, (14, 1) passa em
todas.

Dá −2 níveis nas camadas 4, 5, 6 e 7, −2 nas 1, 2 e 9 se a robustez abaixo de lo for aceitável, e −1 nas outras. Ou
seja, ~17–19 níveis/token contra os 12 do smrel. Todos caem no trecho do q, onde o smrel já mostrou 0 refreshes e
~0 ms. Valor só dentro de um pacote que dê +2 níveis em todos os trechos (tmodel, seção 5).

## 3. H2: Goldschmidt (`h2_frontier.py`, `gs_shift.py`)

**Os alvos intermediários não mexem na precisão final.** O erro de cada rodada intermediária é um fator por linha, e a
normalização seguinte o cancela exatamente. O alvo só define a banda da rodada seguinte, ou seja, a contagem. O
`design()` já escolhe esses alvos na grade para minimizar a soma. O único botão de precisão é o alvo final.

**Achado: o desempate do `design()` deixa precisão de graça na mesa.** Ele guarda o primeiro mínimo da grade (alvo
0,01 antes de 0,2). Nas camadas 0, 2, 6 e 9 sai [7, 4] (alvo 0,01), com resíduo 8,2e-4 em kc = 16. Com o alvo 0,1–0,2
sai [6, 5], com a **mesma** soma 11 e resíduo 4,8e-6 a 5,5e-5. Medido no `softmax_ref`: fator 8,2e-4 → 4,8e-6 e
max|Δp| 8,2e-4 → 2,5e-5 na camada 0 (1,3e-5 na 2, 7e-6 na 6, 5,5e-5 na 9). Na camada 4, [6, 2, 3, 5] → [5, 3, 3, 5]
dá a mesma soma, com fator 4,2e-4 → 4,8e-6.

O preço: −1 nível no trecho da rodada 0 (folga 0) e +1 no da rodada 1 (folga 2–3). Para o item 2, ao contrário, o
melhor é a rodada 1 curta. A ILP decide, e por isso vale gerar as duas configs.

Divisões possíveis da rodada 0 para a rodada 1, todas com resíduo ≤ o do recorde (camadas com δ2 = 2):

| camadas | opções |
|---|---|
| 0, 2, 6, 9 | [6, 5] (5e-6), [5, 6] (6e-5), [4, 7] / [5, 7] (3e-5) |
| 3, 5, 7, 10 | [5, 7] (3e-5) |
| 1 | [4, 7] |
| 8 | [7, 7] |

**Fronteira níveis × alvo final** (soma das contagens; recorde a 1e-3):
- camadas 0, 2 e 6: 1e-2 → 10 (−1, erro ×12); de 1e-4 a 1e-5 → 11 (igual); 1e-6 → 12;
- camadas 3, 5, 7, 9 e 10: 3e-2 → 10, 1e-3 → 11, 1e-4/1e-5 → 11–12;
- camada 4: 3e-2 a 1e-2 → 15 (−1), 1e-3 a 1e-5 → 16.

Afrouxar o alvo final tira no máximo 1 nível em algumas camadas, com erro ×10–30. Não vale.

Confirmo o tmodel: o Goldschmidt escalado por Chebyshev é ótimo em profundidade e em erro para a banda dada. O resíduo
T_{2^k}(z)/T_{2^k}(z0) é o polinômio extremal com r(0) = 1, e a profundidade log2 do grau é o mínimo. Uma semente
melhor para a rodada 0 não ganha nada. Sobra a banda, que é o item 5 e a faixa de dados que vocês descartaram.

## 4. H3: δ2 das camadas 4 e 11 (`h3.py`, `noise.py`, `oor.py`, `margin.py`)

**O que impõe o δ2, pelo código (`fit_gpt2_decode16_ref.fit_centred`):** `delta2 = next_pow2(ustar / tau)`, com
**U\* = o máximo dos dados**, e não U = 1,25·U\*. A margem 1,25 não entra no δ2. Na camada 4, U\* = 18,57 e o limite do
δ2 = 4 é U\* ≤ 18,40 (−0,9 %), ou τ ≥ 4,643.

O upstream já testou exatamente isso. O README do recorde (`speedrun/records/2026-10-27_LeanChainDecode`, Disclosures)
diz que "the criterion's 4.6 was kept over 5.0 (one bit of δ2) after a refresh-noise model's KL on 2,048 prompts of C4
validation shard 00000 doubled at 5.0". Entre τ = 4,6 e 5,0, a única camada centrada que muda é a 4. As camadas 8 e 11
também podem ter mudado, porque viram centradas se ficarem mais rasas, e só os dados dizem.

**O τ é um proxy fraco do ruído.** Na réplica com ruído gaussiano absoluto na saída do exp (o SB#1), a amplificação
segue ~δ2·e^{U/δ2}, com U = 1,25·U\*:

| camada | δ2·e^{U/δ2} | medido rms / max |
|---|---|---|
| 0 | 148 | 30–50 / 195 |
| 4 (recorde, δ2 = 8) | 146 | 51 / 293 |
| 10 (pior atual) | 451 | 92–153 / 596 |
| 4 com δ2 = 4, margem 1,25 (= τ 5,0) | 1330 | 469 / 1497 |
| 4 com δ2 = 4, margem 1,1 | 664 | 234 / 746 |
| 4 com δ2 = 4, margem 1,0 | 415 | 147 / 469 |

Na margem 1,25, a variância do ruído da camada 4 sobe ~80×, e o "KL dobrou" do upstream é coerente com isso. Com a
margem 1,0–1,1, fica no nível da camada 10 até 1,5× dela.

Num modelo grosseiro em que o KL do ruído da softmax ∝ Σ (amplificação²) nas camadas, τ = 5,0 dá ×3,3, a margem 1,1
dá ×1,5 e a margem 1,0 dá ×1,2 na parte do KL que vem do ruído da softmax. O KL do harness é ~6–7,5e-6; mesmo dobrado,
fica 100× abaixo do limite.

**Custo e ganho (camada 4):**

| variante | δ1 | GS | profundidade | piso da rodada 0 |
|---|---|---|---|---|
| recorde | 4 | [6, 2, 3, 5] | 33 | 3,4e-3 |
| τ = 5,0 (δ1 automático) | 8 | [8, 2, 5] | 30 | 1,9e-4 |
| δ2 = 4, δ1 = 4, grau 12 (δ0 = 7,9; grau 8 erraria 7,9e-4 na forma) | 4 | [8, 2, 5] | 29 | |
| idem, margem 1,1 | 4 | [6, 3, 5] | **28** | |
| idem, margem 1,0 | 4 | [7, 2, 5] | 28 | |

São 4 rodadas em vez de 5, ou seja, **−1 SparseB**: os trechos [8 | 5–6 | 8] não se fundem. São ~8,2–8,4 ms (−0,5 %),
mais −4 a −5 níveis que valem ~0 ms. Erro do `softmax_ref`: igual ou melhor que o do recorde (fator 5,5e-5 na margem
1,1).

**Risco de divergência** (`oor.py`). Uma chave acima de U estoura (inf) a partir de U + 0,74·δ2. Em unidades de U\*, a
tolerância é:
- recorde: 1,57 U\* na camada 4, 1,42–1,46 U\* nas outras;
- δ2 = 4: 1,16 (margem 1,0), 1,21 (1,05), 1,26 (1,1), 1,31 (1,15) U\*.

Ou seja, a margem pequena troca ruído por chance de NaN. Só a cauda empírica de max u da camada 4 (hold-out, ≥ 10 mil
janelas) diz se 1,26 U\* é seguro.

**Alternativas legítimas:**
- **Offsets por cabeça** (minha sugestão). A máscara de score já é por slot (`_ref_score_mask` escreve −mean por
  slot), então somar U − U_h por cabeça custa 0 nível. Cada cabeça passa a ter z_ref = e^{−U_h/δ2}, e o ruído cai nas
  cabeças com U_h ≪ U. O piso da banda e o δ2 continuam presos à pior cabeça. Isso não muda os níveis, mas pode tornar
  o δ2 = 4 aceitável com uma margem maior. Precisa de U\*_h por cabeça (dados).
- **Outra referência para a pior cabeça.** Basta tirar 0,9 % do U\* para o δ2 = 4 sair pela própria regra. Sem dados
  não sei se existe. O `fit_centred` só testa sink/self/prev. Vale imprimir o top[r, h] da camada 4.
- **Centrar a 11, ou trocar 1 rodada por mais iterações:** ver H4. As rodadas existem por causa do ruído, não da
  divisão: mais iterações não substituem uma rodada.

## 5. H4: camadas 8 e 11 sem centragem

Pelo código: a camada fica com a softmax base "unless the centred one is shallower" (`depth(c) >= depth(base)`
mantém a base, e **empate mantém a base**). As regras mudam por completo com a centragem:

| | base (sem referência) | centrada |
|---|---|---|
| δ2 | next_pow2(M / (2 ln 2885)) = M/15,9, com M a largura da faixa (cap de magnitude) | U\*/4,6 (ruído relativo à soma da cabeça) |
| banda da rodada 0 | dos dados, ×3 | estrutural, [e^{−U/δ2}/16, 1,05] |
| semente do exp | relativa | mista |

- **Camada 8:** a base tem δ2 = 2 (M = 27,9) e profundidade 24 (banda [0,024, 1140], k0 = 8). A centrada só ganharia
  com U\*_8 ≤ 9,2, que daria ~22 níveis. Como não foi escolhida, U\*_8 > 9,2: δ2 = 4, +1 rodada, +1 SparseB. Não há
  ganho.
- **Camada 11:** a base tem δ2 = 8 (M = 97,7) e profundidade 35. A centrada empata ou perde. Se U\*_11 ≤ 18,4, ela
  teria δ2 = 4 e seria escolhida, logo U\*_11 > 18,4. Se ficar perto disso, o mesmo jogo de margem do H3 tiraria
  1 SparseB, mas com o risco do CutMax a jusante.
- O log do `fit_gpt2_decode16_ref.py` imprime `depth X vs Y` e o U de cada camada: rodar com `--layers 8,11` responde
  em minutos, na sua máquina.

## 6. H5 e outras ideias

- **Semente de grau maior na rodada 0:** não ganha. Ver H2: é o polinômio extremal para a profundidade.
- **Banda "dados com guarda":** polinômio preciso em [lo, D_dados] e só limitado em [D_dados, 1,05]. A banda de dados
  pouparia ~1 nível na rodada 0, mas um polinômio genérico de grau 64 sai por Paterson–Stockmeyer/BSGS com +1 nível e
  muitos produtos. Empata. Descartei.
- **δ1 contra o grau:** com δ2 ≥ 4 (camada 4), as quadraturas do exp sobram, porque as rodadas já elevam à 8ª potência.
  É o (14, 0) da camada 4. Nas camadas com δ2 = 2, cada quadratura troca por ~4 graus extras no critério misto.
- **Grau 7 em vez de 8** com o mesmo k1: −2 produtos por camada, mesma profundidade. Pequeno.
- **Trocar o THOR por algo mais raso:** as rodadas δ2 existem pelo ruído do refresh, não pela aproximação. Qualquer
  esquema com a faixa dinâmica e^{U} num ciphertext só herda o mesmo problema.

## 7. Discordâncias com o brief

1. **"se U ≤ 18,4 seria δ2 = 4" e "o U é 1,25 × o máximo":** o δ2 sai de U\* (18,57), não de U (23,2). A margem não
   entra no δ2. O limite é U\* ≤ 18,40, ou τ ≥ 4,643. E o upstream já testou o δ2 = 4 da camada 4 (τ = 5,0) e recusou
   por ruído (KL dobrou), o que não está nos seus documentos.
2. **"o que limita a softmax é a regra do δ2 (ruído do refresh SparseB)":** em parte. A regra usa U\*, mas o ruído
   escala com δ2·e^{1,25·U\*/δ2}. Ela permite amplificação de até ~1256 no limiar do δ2 = 4, contra ≤ 451 hoje.
3. **"k1 = 0: 5 produtos":** são 7 (`bsgs([1.0]*(d+1))` subconta). O smrel tira 0 produtos, não 2. A coluna de produtos
   da `docs/ideias/softmax.md` (e de `sm_seed_weighted.cost`) está otimista em 1–2.
4. **"sem quadratura (−2 níveis por camada centrada)":** só vale com folga na camada 4. Nas camadas 5–7 o equivalente é
   (14, 1), também −2. Nas outras, o k1 = 0 só passa com o Goldschmidt re-dividido, e perde para o recorde fora da faixa
   (abaixo de clip_lo).
5. **H2 "afrouxar os alvos intermediários":** eles não afetam a precisão, e o `design()` já os otimiza. A direção útil
   é a oposta: **apertar** o final na mesma profundidade (item 3).
6. **"+2 níveis em cada trecho tiram ~11 SparseB":** pela contagem de níveis, o SparseB que pode sumir num bloco com
   δ2 = 2 é o do fim da rodada 1. O trecho que precisa encolher é rodada 1 + softmax_v + out_proj + LN2, onde a semente
   do exp não toca. Conferir em `slack_k2_sb_dense` qual SparseB sumiu.
7. Nas camadas 1, 3 e 7 o erro do recorde é 5,5e-5, não ~1e-3. Um orçamento relativo único é 15× mais apertado nelas.

## 8. O que só dá para validar na sua máquina ou na GPU

- **Captura + ILP:**
  - (a) defer_norm por camada sobre o r4, com e sem o Goldschmidt re-dividido;
  - (b) camada 4 com δ2 = 4, margem 1,1 e δ1 = 4 com grau 12;
  - (c) o pacote de sementes (14, 1)/(14, 0) com guarda, para ver se a folga do trecho do q vira drop ou refresh.
- **Dados (gêmeo):**
  - cauda de max u da camada 4 (e da 11, se centrada) no hold-out, contra 1,16–1,31 U\*;
  - mínimo de u contra clip_lo em todas as camadas;
  - U\*_h por cabeça e top[r, h] com outras referências;
  - KL no gêmeo dos pacotes.
- **GPU:** o KL cifrado (o modelo de ruído do upstream, ou o harness) da camada 4 com δ2 = 4, e o tempo.

## Arquivos

Todos em `subagent/softmax2/scripts/`. Rodar com
`PYTHONPATH=<clone>/python:.` e o python do venv:

| arquivo | o que faz |
|---|---|
| `sm2.py` | harness, LP do critério misto, `fit_lp_ext` com guarda |
| `seed_search.py`, `seed_search.log` | H1 por camada |
| `pacote.py`, `pacote.log` | semente + GS, fora da faixa, bandas, ruído |
| `ext.py`, `ext1.py`, `ext.log` | guarda abaixo de lo |
| `h2_frontier.py`, `gs_shift.py` | H2 |
| `h3.py`, `noise.py`, `oor.py`, `margin.py` | H3 e a margem |
| `seg_slack.py`, `trace_sm.py` | níveis e pousos por refresh, do grafo e do plano do recorde |

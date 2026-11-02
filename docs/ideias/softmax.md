# Softmax do recorde 4: onde ainda há níveis

> **2026-11-02 — o upstream absorveu o candidato 1:** o recorde 7 já roda a camada 4 com δ2 = 4 (commit 1d945f2, plano
> `_lhs2_pm`: "the softmax of layers 4 and 11 in four SparseB refreshes instead of five"). Ele também dobrou os scores
> no Goldschmidt (`gs_num`: um produto e um nível a menos por rodada), pareou cabeças (SparseB-256) e refez a exp com grau
> 6, com o peso p^(2^s − 1). Os candidatos 2-4 abaixo (normalização adiada, Goldschmidt re-dividido, semente pelo
> critério misto) foram medidos sobre o recorde 4 e precisam ser refeitos sobre o recorde 7 antes de qualquer conta.

Data: 2026-11-01. Status: **em aberto, com 4 candidatos que só a ILP e o KL decidem.** Nada sozinho chega aos 1 % da
regra 5. Nada rodou em GPU. Análise completa: `subagent/softmax2/relatorio.md`, a segunda opinião de um subagente Opus
medida com o `softmax_ref` real do upstream (`SM_CHEB_GS=1`), linhas sintéticas e réplica com ruído no refresh.

O título antigo desta nota ("sementes compostas e aproximação ponderada") vinha de uma proposta que usava erro absoluto
puro e o (d=3, k=3). Esse caso tem erro relativo 8,9e-2 e não serve. O que sobrou da ideia está na seção 3.

## 1. Estrutura (recorde 4, `gpt2_base_n32_kc16r_cm`)

THOR: `z = P(t)^δ1 ≈ e^{(u−U)/δ2}`. Depois a rodada 0 normaliza (`y = z / Σz`, Goldschmidt), e cada rodada de refino
eleva ao quadrado e normaliza de novo, log2(δ2) vezes. Com δ2 = 2, cada camada tem 3 SparseB: na saída do exp, na saída
da rodada 0 e depois da última rodada.

| camadas | centrada | δ1 | δ2 | Goldschmidt por rodada | fator por linha (erro real) |
|---|---|---|---|---|---|
| 0, 2, 6, 9 | sim | 4 (6: 8) | 2 | [7, 4] | 8,2e-4 |
| 1, 3, 7 | sim | 4 (7: 8) | 2 | [5, 5], [6, 5], [6, 5] | 5,5e-5 |
| 5, 10 | sim | 8, 4 | 2 | [6, 5] | 4,2e-4 |
| **4** | sim | 8 | **8** | [6, 2, 3, 5] | 4,2e-4 |
| 8 | não | 4 | 2 | [8, 5] | — |
| **11** | não | 4 | **8** | [7, 3, 3, 5] | — |

- **O erro real é dominado pelo Goldschmidt:** o fator por linha da última rodada (alvo 1e-3) fica 15 a 185× acima do
  erro de forma do exp, e varia 15× entre camadas.
- **O δ2 vem de `next_pow2(U*/τ)`** (`fit_gpt2_decode16_ref.fit_centred`, τ = 4,6), com U\* o máximo dos dados por
  cabeça. **Não** vem do U = 1,25·U\*, como eu tinha escrito. Na camada 4, U\* = 18,57 fica a 0,9 % do limiar de δ2 = 4
  (18,40).
- **O upstream já testou τ = 5,0**, que só muda a camada 4, e recusou: o KL de um modelo de ruído dobrou (README do
  `2026-10-27_LeanChainDecode`, linha 241).

## 2. Candidatos (ranking do subagente, conferido onde indicado)

| # | ideia | níveis / refreshes | ms (tmodel) | risco | próximo passo |
|---|---|---|---|---|---|
| 1 | **δ2 8 → 4 na camada 4 com margem 1,0–1,1** (em vez de τ = 5 com margem 1,25). **Testado na CPU:** `docs/testes/softmax-camada4.md` | −1 SparseB, −4 a −5 níveis | ≈ −8,4 | ruído do 1º SparseB da camada 4 ×2,8–4,5 (rms; τ = 5 dava ×9); tolerância acima de U\* cai de 1,57 para 1,16–1,26 U\*, e acima disso o Goldschmidt diverge (NaN) | cauda de max u da camada 4 no hold-out, KL do modelo de ruído, ILP |
| 2 | **normalização adiada (`defer_norm`) sobre o recorde 4** | −2 níveis por camada no trecho rodada 1 → softmax_v → out_proj → LN2, onde o 3º SparseB pode sumir (13 → ≤ 11 níveis) | 0 a −8,4 por bloco; teto ~−90 | no recorde 3, adiar na camada 11 deixou o CutMax inviável; no recorde 4 nunca foi para a ILP | portar o branch `defer-norm` e rodar a ILP camada a camada |
| 3 | **Goldschmidt re-dividido, mesma profundidade**: [7, 4] → [6, 5] nas camadas 0, 2, 6, 9; [6, 2, 3, 5] → [5, 3, 3, 5] na 4 | 0 no total (−1 na rodada 0, +1 na 1) | 0 sozinho | o fator por linha cai 100–170× (8,2e-4 → 4,8e-6); mas gasta folga da rodada 1, que é o trecho do item 2 | dar as duas configs à ILP |
| 4 | **semente do exp pelo critério misto** (seção 3) | ≈ −17 a −19 níveis/token (smrel: −12) | ≈ 0 (cai no trecho do q, como o smrel) | abaixo de clip_lo o (14, 0) sem guarda erra 0,6–0,9; o (14, 1) precisa ser ajustado já com o `CHEB_COEFF_EPS` | só dentro de um pacote de folga, com o item 2 |
| 5 | margem superior 1,25 → 1,1 nas camadas centradas | −1 nível nas camadas 0, 2, 4, 6, 9 | ≈ 0 | tolerância acima de U\* cai de ~1,45 para ~1,3 U\* | só com a cauda dos dados |

Descartados:
- **Afrouxar os alvos intermediários do Goldschmidt:** não muda a precisão (o fator de cada rodada some na normalização
  seguinte), e o `design()` já os escolhe para minimizar níveis.
- **Afrouxar o alvo final para 1e-2:** −1 nível em 4 camadas, com erro ×12.
- **Semente melhor na rodada 0:** o Goldschmidt escalado por Chebyshev já é ótimo em profundidade e erro para a banda
  dada.

Camadas 8 e 11: o script de ajuste só centra quando fica mais raso, e no empate mantém a base. Na 8, centrar
acrescentaria uma rodada. A 11 só ganha se o U\* dela ficar perto de 18,4.
`fit_gpt2_decode16_ref.py --layers 8,11` na máquina do usuário responde em minutos.

## 3. A semente do exp por bandas (o que sobrou da proposta original)

A restrição `L(x) ≤ P(x/n) ≤ U(x)` é afim nos coeficientes de Chebyshev, e o melhor ε sai por bissecção com LP
(ATAP, cap. 10). A métrica certa depende da camada:

- **Sem chave de referência** (camadas 8 e 11): erro relativo, `(1±ε)^(1/n)`, que é o minimax relativo do `smrel`.
- **Centradas:** a referência está sempre ativa na linha com u = 0 (sink, self ou prev, sempre em `range(kc)`), então
  Σe^{u_j} ≥ 1 e p_i ≤ e^{u_i}. O erro entra como `|Ê − e^u| / max(1, e^u)`: absoluto para u ≤ 0, relativo para u > 0
  (`scripts/sm_seed_weighted.py`).

Com o alvo honesto (erro total ≤ o do recorde, por camada):
- **sem quadratura (k1 = 0):** só a camada 4 passa limpa; as camadas 0, 2, 9 e 10 passam só somadas ao item 3;
- **(14, 1):** passa em todas, e nas camadas 5–7 tira 2 quadraturas.

Minha primeira busca (`scripts/sm_seed_search.py`) usou como alvo o erro de forma, que é 15 a 185× mais estrito, e por
isso concluiu errado que "sem quadratura não dá em nenhuma camada".

**Custo real da semente** (BSGS com coeficientes genéricos; as minhas tabelas antigas usavam `[1.0]*(d+1)`, que
subconta 1–2 produtos; `sm_seed_weighted.cost` foi corrigido):

| grau | 5–6 | 7 | 8 | 9–11 | 12–14 | 15 |
|---|---|---|---|---|---|---|
| (níveis, ct×ct) | (3, 4) | (4, 4) | (4, 6) | (4, 6) | (4, 7) | (5, 7) |

Então o recorde (8, 2) custa (6, 8), o smrel (14, 1) custa (5, 8) e o (14, 0) custa (4, 7). A semente tira níveis, mas
quase nenhum produto.

## 4. O que precisa da máquina do usuário ou da GPU

- **ILP:**
  - a camada 4 com δ2 = 4;
  - a normalização adiada no recorde 4;
  - o pacote de sementes e a re-divisão do Goldschmidt;
  - conferir em `slack_k2_sb_dense` qual dos três SparseB sumiu em cada bloco.
- **Gêmeo com dados reais:**
  - as caudas de u contra clip_hi e clip_lo;
  - o U\* de cada cabeça. Um offset por cabeça na máscara de score não custa nível e reduziria o ruído das cabeças com
    U\* menor.
- **GPU:** o KL cifrado e o tempo.

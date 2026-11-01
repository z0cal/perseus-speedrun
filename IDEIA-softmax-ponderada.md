# Ideia: semente do exp da softmax ajustada à cadeia de quadraturas (bandas)

Data: 2026-11-01. Status: **fechada. Com o critério certo, a melhor semente cai na vizinhança do `smrel` (uma
quadratura a menos), que já foi medido.** O ganho sobre o smrel é ≤ 1 produto por camada. Nada rodou em GPU.

## Contexto

A fase exponencial da softmax (`_thor_exp`) é *scaling and squaring*: `H(x) = P_d(x/2^k)^(2^k)`. No recorde 4
(`gpt2_base_n32_kc16r_cm`) a semente tem grau `d = 8` em `|y| ≤ δ0/2` (camada 0: δ0/2 = 2,43) e `k = 2` ou 3
quadraturas, conforme a camada. A divisão usa Goldschmidt com `gs_iters` 6–7 na primeira rodada e 4–5 no refino.

## A formulação por bandas

A proposta original (sementes compostas) restringe a semente a uma banda que já leva em conta as quadraturas:

`sup |P(x/n)^n − e^x| ≤ ε  ⇔  L(x) ≤ P(x/n) ≤ U(x)`, com `n = 2^k` e `P ≥ 0`.

É afim nos coeficientes de Chebyshev, então o conjunto viável é convexo e o melhor ε sai por bissecção com LP. O
espírito é o mesmo do minimax ponderado de `scripts/em_fit.py` (ATAP, cap. 10).

### Qual métrica

A versão original usava erro **absoluto** (`L = max(e^x − ε, 0)^(1/n)`, `U = (e^x + ε)^(1/n)`), com o argumento de que
a cauda não importa. A resposta depende da camada:

- **Sem chave de referência** (recorde 3; camadas 8 e 11 do recorde 4): a linha pode ter todos os scores no fundo da
  faixa, então o erro tem de ser **relativo**. O ajuste absoluto do exp largo já foi medido e estourou (KL 3,1e-4;
  `subagent/ganhos-rapidos.md`, item 3). Com bandas relativas, `(1−ε)^(1/n) ≤ P/e^y ≤ (1+ε)^(1/n)`, a restrição vira
  o minimax relativo da semente: é o que o `smrel` (`subagent/search2.py::apply_smrel`) já faz.
- **Centradas** (as outras 10 camadas do recorde 4): a chave de referência está na linha com u = 0, então
  `Σ_j e^{u_j} ≥ 1` e `p_i ≤ e^{u_i}`. O erro do exp entra em p como `|Ê(u) − e^u| / max(1, e^u)`: **absoluto para
  u ≤ 0, relativo para u > 0**. As rodadas δ2 são potências exatas dos valores normalizados e não mudam isso em
  primeira ordem. É o critério misto do `scripts/sm_seed_weighted.py` (bandas LP com esse peso).

### O critério misto ganha algo? Pouco

O cap. 8 do ATAP sugere que, com erro absoluto, o exp num intervalo longo dispensa o *scaling and squaring*. Testei isso
na cadeia completa simulada (`scripts/sm_seed_sim.py` e `sm_seed_search.py`). Usei linhas sintéticas de 16 chaves, com
a referência em u = 0, metade uniformes e metade agrupadas, e aritmética exata. O critério foi o max|p − p_exato| do
recorde em cada camada:

| camada (recorde) | recorde: níveis, ct×ct | k1 = 0 (sem quadratura) | k1 − 1 | mesmo k1 |
|---|---|---|---|---|
| 0 (d8, k1=2) | 6, 8 | não alcança (d ≤ 15) | d11: 5, 7 | d7: 6, 6 |
| 2 (d8, k1=2) | 6, 8 | não alcança | d10: 5, 6 | d8: 6, 8 |
| 9 (d8, k1=2) | 6, 8 | não alcança | d11: 5, 7 | d8: 6, 8 |
| 5 (d8, k1=3) | 7, 9 | não alcança | d9 (k1=2): 6, 7; d13 com k1=1: 5, 6 (−2 níveis) | d7: 7, 7 |

- Sem quadratura não dá: o erro absoluto equioscila na faixa toda, e a normalização soma o erro das 16 chaves.
  Na primeira tentativa (grau 14, 4 níveis), max|Δp| deu 4,3e-4 contra 2,3e-5 do recorde na camada 0, e a camada 11
  (sem referência) falhou por completo.
- Com uma quadratura a menos, o critério misto alcança o recorde com grau 10–13, contra 14 do smrel: o mesmo número de
  níveis e 0 a 1 produto a menos. Não muda a conclusão do smrel, que foi medido com 0 refreshes a menos.
- Ressalvas: linhas sintéticas (não capturadas), sem ruído CKKS. O Σ|c| das sementes do LP cresce com a faixa, então o
  condicionamento precisaria de um teste no gêmeo antes de qualquer uso.

## Fronteira (níveis × produtos × erro relativo)

Mesma faixa total do recorde na camada 0 (`h = 2,43·2^(2−k)`). P é o minimax relativo por LP. Os produtos e a
profundidade vêm de `scripts/gelu_bsgs_count.bsgs` (BSGS do upstream, coeficientes densos, sem contar o mapa afim, que é
igual em todas as linhas) mais as k quadraturas.

| semente (d, k) | erro relativo de P^(2^k) | níveis | ct×ct |
|---|---|---|---|
| **recorde (8, 2)**, refeito como minimax relativo | 1,1e-4 | 6 | 8 |
| (3, 3), o "cenário B" da proposta original | **8,9e-2** (inviável) | 5 | 5 |
| (6, 3) | 9,3e-5 | 6 | 6 |
| (9, 2) | 1,4e-5 | 6 | 7 |
| (13, 1) | 8,0e-6 | 5 | 6 |
| **(14, 1) = `smrel` r=1** | ≤ 8e-6 | **5** | **6** |

- O cenário (d=3, k=3) só parecia bom no erro absoluto: no relativo ele é ~800× pior que o recorde.
- O melhor ponto é o que o `smrel` já usa: 1 nível e 2 produtos a menos por camada, e erro relativo ~14× menor. No
  recorde 4 isso foi medido: −12 níveis/token, KL neutro, mas **0 refreshes a menos** e ~0 ms no `tmodel`. A folga de
  nível não vira refresh, e 2 produtos por camada pesam pouco (a softmax inteira são 74 ms/token).
- (6, 3) e (9, 2) mantêm a profundidade e tiram 2 ou 1 produtos. São só alternativas ao smrel quando o nível a mais não
  serve para nada.

## Goldschmidt

A proposta original sugeria cortar `gs_iters` de 8 para 6 graças a "denominadores mais estáveis". Isso não procede:

- o recorde 4 já usa 6–7 e 4–5 iterações, não 8;
- o `tmodel` (seção 5) mostrou que o Goldschmidt escalado por Chebyshev **já é** o minimax relativo de 1/x (o resíduo é
  T_{2^k}(·)/T_{2^k}(z0)), e que a contagem de iterações vem da faixa do denominador e da regra do δ2 (ruído do refresh
  SparseB), não da semente do exp.

## Conclusão

Nada a implementar além do `smrel`, que já existe. Se ele entrar num pacote, use o grau 13 ou 14 (4 níveis no BSGS).
O grau 16 custaria 5 níveis e 10 produtos.

Reproduzir a tabela: LP relativo de `e^y` em `[−h, h]` (2000 pontos de Chebyshev, `scipy.optimize.linprog`, HiGHS) e
contagem por `scripts/gelu_bsgs_count.bsgs([1.0]*(d+1))`.

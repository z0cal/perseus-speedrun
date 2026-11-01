# Ideia: semente do exp da softmax ajustada à cadeia de quadraturas (bandas)

Data: 2026-11-01. Status: **fechada. Feita do jeito certo (erro relativo), a ideia vira o `smrel`, que já foi medido.**
Nada rodou em GPU.

## Contexto

A fase exponencial da softmax (`_thor_exp`) é *scaling and squaring*: `H(x) = P_d(x/2^k)^(2^k)`. No recorde 4
(`gpt2_base_n32_kc16r_cm`) a semente tem grau `d = 8` em `|y| ≤ δ0/2` (camada 0: δ0/2 = 2,43) e `k = 2` ou 3
quadraturas, conforme a camada. A divisão usa Goldschmidt com `gs_iters` 6–7 na primeira rodada e 4–5 no refino.

## A formulação por bandas

A proposta original (sementes compostas) restringe a semente a uma banda que já leva em conta as quadraturas:

`sup |P(x/n)^n − e^x| ≤ ε  ⇔  L(x) ≤ P(x/n) ≤ U(x)`, com `n = 2^k` e `P ≥ 0`.

É afim nos coeficientes de Chebyshev, então o conjunto viável é convexo e o melhor ε sai por bissecção com LP. O
espírito é o mesmo do minimax ponderado de `scripts/em_fit.py` (ATAP, cap. 10).

### A métrica tem de ser relativa

A versão original usava erro **absoluto** (`L = max(e^x − ε, 0)^(1/n)`, `U = (e^x + ε)^(1/n)`), com o argumento de que
a cauda não importa. Isso não vale aqui:

- o ajuste absoluto do exp largo já foi medido e **estourou** (KL 3,1e-4; `subagent/ganhos-rapidos.md`, item 3);
- a softmax é invariante a deslocamento por linha, e a do recorde 4 é centrada numa chave de referência, não no máximo
  da linha. Uma linha com todos os scores no fundo da faixa precisa de precisão **relativa** lá.

Com bandas relativas, `(1−ε)^(1/n) ≤ P(y)/e^y ≤ (1+ε)^(1/n)`, as bandas ficam quase simétricas, com largura ≈ ε/n. A
restrição vira exatamente o **minimax relativo da semente** com tolerância ε/n: o mesmo problema que o IRLS relativo do
`smrel` (`subagent/search2.py::apply_smrel`) resolve. A teoria das bandas não acrescenta nada além disso.

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

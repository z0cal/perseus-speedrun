# Ideia: GELU com série PAR (Trefethen, *Approximation Theory and Approximation Practice*, caps. 3–4)

> **2026-11-02, portada para o recorde 7 (README, seções de 2026-11-02):** `poly.eval_chebyshev_even` com metade dos
> babies (mesma profundidade em todos os blocos), configs `_ge` (todos os blocos, ímpares pequenos zerados) e `_ge48`
> (blocos 4-8, função idêntica). No recorde 7 (GELU grau 60/222 com peso): 212 → 178 produtos/token. Modelo R7: `_ge`
> −15,8 ms (−1,6 %), quase tudo por zerar os ímpares (−13,7 ms); `_ge48` −0,7 ms. Falta captura + plano + KL.

Data: 2026-11-01. Status: **contada, ganho pequeno (~0,2 % do token); prioridade baixa.** Nada rodou em GPU.

## Resumo

- A GELU decompõe-se em `GELU(x) = 0,5·x + (x/2)·erf(x/√2)`. O segundo termo é **par**: o produto de duas funções ímpares.
- Nas configs do Perseus a série de Chebyshev está em `[−1, 1]` (`cheb_a = −1`, `cheb_b = 1`, após dividir por
  `X = 1,3·xmax`). Por isso `c_1 = 0,5` e os coeficientes ímpares de grau ≥ 3 valem ~1e-16.
- Com `T_2k(x) = T_k(2x² − 1)` (ATAP, cap. 3), a parte par vira uma série de **metade do grau** em `z = 2x² − 1`.
- **O BSGS atual já aproveita quase tudo isso.** O `_cheb_trim` zera os ímpares (|c| ≤ 1e-9·max|c|), e o
  `IMPL_DEAD_OPS`, ligado por padrão, pula os babies ímpares que nenhuma folha lê. O ganho que sobra é
  **210 → 180 produtos ct×ct por token (−14 %)**, cerca de **−4 ms (0,2 %)**.

## Contagem (config do recorde 4, `gpt2_base_n32_kc16r_cm`)

Script: `scripts/gelu_bsgs_count.py`. Ele reimplementa `eval_chebyshev_bsgs` do upstream d97a6f7 (babies, giants,
`_cheb_needed`, divisão recursiva) sem importar o Perseus. Reproduz a tabela de profundidade de
`subagent/ganhos-rapidos/ganhos-rapidos.md`: 60 → 6, 62 → 7, 248 → 8 e 254 → 9 níveis.

| grau aparado (camadas) | atual: níveis, ct×ct | sem dead_ops | par em z, mesmo nível | 1 nível a menos |
|---|---|---|---|---|
| 62 (0, 4, 5, 6, 8, 9) | 7, 14 | 7, 16 | 7, 12 | 6, 20 (em z, ou em x com m=2) |
| 60 (7) | 6, 14 | 6, 16 | 6, 13 | — |
| 94 (1, 3, 10) | 7, 18 | 7, 22 | 7, 15 | — |
| 254 (2, 11) | 9, 29 | 9, 33 | 9, 25 | 8, 70 |
| **total/token** | **210** | | **180** | |

O `tmodel` estima ~26 ms de produtos e key switches na GELU por token (42 ms ao todo). Escalando, −30 produtos dão
**≈ −4 ms**. É pouco demais para justificar a mudança sozinha; só vale dentro de um pacote com outras.

### Achado lateral: um nível sem cortar grau

O grau 62 cai para 6 níveis com +6 produtos (m=2 no BSGS em x, ou a forma em z), e o grau 254 cai para 8 com +41.
O `gelubs` conseguia o mesmo nível cortando o grau (62 → 60, 254 → 248), mas o KL subia (2,7e-6 → 4,1e-6). Só vale se
esse nível virar refresh a menos. No recorde 4, o `subagent/r4/relatorio.md` diz que o gelubs "não compensa".

## Formulação

1. `z = 2x² − 1`: **1 square + `add(sq, sq)` + `add(·, −1)`**. O ×2 é uma soma e não gasta nível; é o mesmo que o BSGS
   já faz em `T[i] = ops.add(ops.add(sq, sq), -1.0)`. Não precisa absorver √2 na matriz `Up`.
2. `P(z) = Σ_k c_{2k} T_k(z)`, avaliada com o mesmo `eval_chebyshev_bsgs` (`a = −1`, `b = 1`, entrada no nível 1).
3. `GELU(x) = 0,5·x + P(z)`. O `0,5·x` desce até o nível de `P` (`level_drop`) antes de somar.

Faixa: `x ∈ [−1, 1] ⇒ z ∈ [−1, 1]`. Se |x| > 1, então z > 1, e a série cresce fora da faixa como já cresce hoje em x.

## Como implementar (se for entrar num pacote)

1. Em `python/perseus/impl/poly.py`: quando os ímpares de grau ≥ 3 aparados forem 0, avaliar a parte par em z e somar
   `c_1·x`; senão, cair no BSGS atual.
2. Teste CPU comparando com o BSGS atual em dados sintéticos (tolerância < 1e-6) e o KL do gêmeo float64.
3. Recontar o plano: a profundidade não muda na escolha "mesmo nível", então os refreshes e pousos ficam iguais.
4. Medir na GPU só dentro do A/B de um pacote; sozinho, o ganho fica abaixo do ruído.

# Ideia: GELU com série PAR (Trefethen, *Approximation Theory and Approximation Practice*, Caps. 3–4)

Data: 2026-11-01. Status: **revisada no mesmo dia: o ganho real é ~7× menor que o previsto abaixo.** Prioridade baixa.

> **Revisão 2026-11-01 (contagem real, `scripts/gelu_bsgs_count.py` na config do recorde 4).** O `IMPL_DEAD_OPS`, ligado
> por padrão, já pula os babies ímpares mortos: os coeficientes ímpares de grau ≥ 3 viram 0 exato no `_cheb_trim`. O
> BSGS atual faz **14** produtos no grau 62, não 26. A forma par em z faz 12 na mesma profundidade. No token inteiro são
> **210 → 180 produtos (−30, não −170)**, ou ~−4 ms (0,2 %). Os graus do recorde 4 são 62/94/254 (60 na camada 7), não 63.
> Profundidade: 62 → 7 níveis e 254 → 9, iguais em z; z também permite 62 → 6 níveis com +6 produtos (o mesmo que m=2 em x).
> O ×2 de z = 2x² − 1 é `add(sq, sq)`, sem nível, como o próprio BSGS já faz nos babies: a seção 5.1 não se aplica.
> Detalhes e tabela: `ATAP-ganhos.md`, seção 3. O texto original segue abaixo, sem edição, para registro.

## 1. Resumo

No Perseus (GPT-2 Decode-8), a GELU é avaliada como uma série de Chebyshev de grau 63 via agendamento BSGS (`eval_chebyshev_bsgs` em `python/perseus/impl/poly.py`).

A GELU decompõe-se analiticamente em:
$$\text{GELU}(x) = 0.5x + \frac{x}{2} \text{erf}\left(\frac{x}{\sqrt{2}}\right)$$

Como $x$ é ímpar e $\text{erf}(x)$ é ímpar, o produto $\frac{x}{2} \text{erf}\left(\frac{x}{\sqrt{2}}\right)$ é **estritamente PAR**. 
Nos coeficientes reais do `configs.json` do Perseus (ex.: `r4_ln3`), $c_1 = 0,5$ e todos os coeficientes ímpares para $k \ge 3$ são rigorosamente nulos ($|c_k| < 10^{-15}$).

Aplicando a identidade clássica de paridade de Chebyshev (Trefethen, Cap. 3–4):
$$T_{2k}(x) = T_k(2x^2 - 1)$$

Definindo $z = 2x^2 - 1$ (1 elevação ao quadrado no início), a série par residual passa a ter **metade do grau: grau 31 em $z$**:
$$\text{GELU}(x) = 0.5x + \sum_{k=0}^{31} c_{2k} T_k(z)$$

**Resultado previsto:**
* Redução de multiplicações de ciphertext de **~26 para 12 por GELU** (~14 multiplicações a menos por camada, **~170 multiplicações com key-switching a menos no token de 12 camadas**).
* **Profundidade multiplicativa inalterada (empate em 7 níveis)**: o plano global de bootstraps e patamares de pouso não é afetado.

---

## 2. Contexto e Correção

* Em `IDEIA-evalmod-par.md` (2026-10-30), havia sido anotado que *"GELU e softmax não ganham com o livro"*. Essa afirmação desconsiderou que a GELU no Decode-8 roda como Chebyshev direto de grau 63, cujos termos ímpares para $k \ge 3$ são nulos.
* No modelo de tempo calibrado (`tmodel/RELATORIO-2026-10-30.md`), a etapa de GELU consome **~42 ms por token** (26 ms de produtos de ciphertext + key-switches). Economizar ~170 multiplicações de ciphertext ataca diretamente essa fatia.

---

## 3. Formulação Matemática

Para $x \in [-1, 1]$ (após normalização por $x_{\max}$):

1. **Transformação de Paridade:**
   $$z = 2x^2 - 1 \quad \in [-1, 1]$$
   Para evitar uma multiplicação escalar extra por $2$ (que consumiria 1 nível homomórfico ou rescale), absorve-se $\sqrt{2}$ na normalização da projeção linear anterior (`Up`):
   $$z = (\sqrt{2}x)^2 - 1$$
   Dessa forma, o cálculo de $z$ custa **estritamente 1 square de ciphertext + 1 adição de constante $(-1)$**.

2. **Avaliação da Série em $z$:**
   $$P_{\text{even}}(z) = \sum_{k=0}^{31} c_{2k} T_k(z)$$
   Avaliada via BSGS com $m = 4$ ($3$ babies $T_1..T_3$, e giants até $T_{16}$): consome **11 multiplicações de ciphertext**.

3. **Recombinação Linear:**
   $$\text{GELU}(x) = 0.5x + P_{\text{even}}(z)$$
   O termo linear $0.5x$ sofre `level_drop` para casar o nível RNS com a saída de $P_{\text{even}}(z)$ e é somado.

---

## 4. Parecer de um agente (2026-11-01, sem contagem; ver a revisão no topo)

A proposta foi submetida ao agente Claude Sonnet para análise crítica em 2026-11-01. O parecer técnico concluiu:

### A. Matemática e Precisão
* A matemática é rigorosamente correta.
* Em CKKS, a troca de ordem das operações introduz leves variações de arredondamento em relação ao grau 63 direto, mas a precisão permanece na ordem de $\le 10^{-6}$ (mesma fidelidade do modelo).

### B. Profundidade Multiplicativa e Níveis (Sem ganho de nível)
* **A proposta não economiza níveis:**
  * Grau 63 no BSGS: $\lceil \log_2(63) \rceil + 1 = 6 + 1 = \mathbf{7\text{ níveis}}$.
  * Grau 31 no BSGS: $\lceil \log_2(31) \rceil + 1 = 5 + 1 = 6\text{ níveis}$.
  * Com o squaring inicial de $z$ ($+1\text{ nível}$): $1 + 6 = \mathbf{7\text{ níveis}}$.
* **Conclusão:** A profundidade é idêntica. Não há eliminação de bootstraps nem alteração no ILP, mas há redução expressiva no volume de computação na GPU.

### C. Latência na GPU
* Redução de ~14 multiplicações com key-switch por camada (~170 no decode completo).
* Como a GELU totaliza ~42 ms no decode, a redução de computação é real, embora o ganho absoluto em ms deva ser modesto (estimado em ~5 a ~10 ms). Precisa ser validado por medição A/B na GPU.

---

## 5. Riscos e Armadilhas de Implementação

1. **Evitar multiplicação escalar extra em $z$:**
   * Não calcular `mult(square(x), 2.0) - 1.0`.
   * Absorver $\sqrt{2}$ na matriz `Up` ou usar a constante dobrada na relinearização do squaring.
2. **Alinhamento de níveis no termo linear ($0.5x$):**
   * O $x$ entra no nível do topo e precisa descer 6 níveis (`level_drop`) antes da adição final com $P_{\text{even}}(z)$.
3. **Estabilidade de faixa:**
   * Se $x \in [-1, 1]$, então $z \in [-1, 1]$. O squaring não extrapola o domínio $[-1, 1]$.
   * Se $x$ já chegar com ruído fora de $[-1, 1]$, o squaring pode amplificar levemente o desvio nas bordas; verificar se $x_{\max}$ tem margem adequada.
4. **Generalidade e Fallback:**
   * Caso alguma configuração use aproximação sem simetria par (ex.: aproximações com termos ímpares em $k \ge 3$), a função deve detectar a paridade dos coeficientes e recorrer ao BSGS convencional de grau ímpar.

---

## 6. Como Implementar

1. **Em `python/perseus/impl/poly.py`:**
   Adicionar a variante `eval_chebyshev_even(ops, x, coeffs, a, b, eps)` ou chavear dinamicamente dentro de `eval_chebyshev_bsgs` quando os coeficientes ímpares $c_3, c_5, \dots$ forem nulos:
   ```python
   # Se c_k == 0 para todo k ímpar >= 3:
   # 1. z = square(x * sqrt(2)) - 1.0 (ou square + add + constant)
   # 2. q = eval_chebyshev_bsgs(ops, z, coeffs_even, -1.0, 1.0)
   # 3. return ops.add(q, ops.level_drop(ops.mult(x, coeffs[1]), target_level=q.level))
   ```
2. **Testes Unitários:**
   * Teste em `python/tests/` comparando a saída de `eval_chebyshev_even` com `eval_chebyshev_bsgs` em dados sintéticos (tolerância $< 10^{-6}$).
3. **Medição na GPU:**
   * Rodar o A/B do Decode-8 medindo o delta de tempo no kernel de GELU e no tempo final por token.

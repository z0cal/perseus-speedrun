# Ideia de Otimização: Sementes Compostas e Aproximação Ponderada para a Softmax no Perseus

Este documento formaliza a aplicação da teoria de sementes compostas e bandas de erro exatas (desenvolvida em `/home/zocal/pesquisa/exp_approximation_design/theory/COMPOSED_SEEDS.md` e `THEORY.md`) para otimizar a fase exponencial da Softmax homomórfica no Perseus Speedrun (GPT-2 Decode).

---

## 1. Contexto no Perseus Speedrun

Na arquitetura atual do Perseus (`work/perseus/python/perseus/impl/attention.py` e `work/perseus/python/perseus/calibrate/approximations/softmax.py`):
- A Softmax consome ~74 ms e exige até 40 refreshes de bootstrap SparseB-512 por token.
- A fase exponencial (`_thor_exp`) implementa o paradigma *scaling-and-squaring*:
  $$H_{d, k}(x) = \left( P_d\left(\frac{x}{2^k}\right) \right)^{2^k}$$
  onde avalia-se um polinômio de Chebyshev $P_d$ (grau padrão $d=8$) e aplica-se $k = \log_2(\delta_1)$ elevações ao quadrado (`ops.square`).
- **A Ineficiência Atual:**
  Os coeficientes em `cheb_exp_series` (`numerics.py`) são calculados via ajuste uniforme clássico de Chebyshev para $e^y$ no subintervalo $[-0.5\delta_0, 0.5\delta_0]$, normalizando $P(0)=1$.
  Esse ajuste **ignora a cadeia de $k$ squarings subsequente**, gerando:
  1. **Amplificação assimétrica de erro:** Um erro residual $\delta$ na semente é multiplicado por $2^k e^{(1 - 2^{-k})x}$. Perto de $x=0$, a amplificação é de $2^k$, enquanto na cauda negativa ($x \approx -32$) o erro é atenuado a $10^{-12}$.
  2. **Desperdício de graus de liberdade:** O polinômio gasta esforço tentando atingir precisão extrema na cauda negativa profunda, onde o valor de $e^x$ é desprezível para a Softmax, enquanto perde precisão crítica perto de $x=0$.
  3. **Ruído residual nos denominadores:** Isso obriga o uso de 8 iterações de divisão de Goldschmidt (`gs_iters: 8`).

---

## 2. A Teoria das Bandas Exatas (Composed Seeds)

A teoria desenvolvida em `COMPOSED_SEEDS.md` resolve exatamente esse problema:

### O Teorema das Bandas Exatas
Para qualquer intervalo $I = [a, 0]$, potência $n = 2^k$ e tolerância máxima de erro $\epsilon > 0$, e sob a restrição de semente não-negativa $P(y) \ge 0$:
$$\sup_{x \in I} |P(x/n)^n - e^x| \le \epsilon \iff L_\epsilon(x) \le P(x/n) \le U_\epsilon(x) \quad \forall x \in I$$
onde:
$$L_\epsilon(x) = \max(e^x - \epsilon, 0)^{1/n}$$
$$U_\epsilon(x) = (e^x + \epsilon)^{1/n}$$

### Propriedades Analíticas Fundamentais
1. **Convexidade das Restrições:** Para qualquer ponto $x$, $P(x/n) = \sum_{j=0}^d c_j T_j(x/n)$ é afim nos coeficientes de Chebyshev $c$. As desigualdades $L_\epsilon(x) \le \sum c_j T_j \le U_\epsilon(x)$ formam um conjunto convexo fechado de coeficientes viáveis (provado formalmente em Lean 4 em `ConvexBands.lean`).
2. **Quasiconvexidade do Erro Global:** O erro final de aproximação é quasiconvexo, garantindo a ausência de mínimos locais espúrios e permitindo encontrar o polinômio ótimo global via **bissecção simples com Programação Linear (LP)** via solver HiGHS (`linprog`).
3. **Ponderação e Relaxamento Automáticos:**
   - Onde $e^x \le \epsilon$, $L_\epsilon(x) = 0$: a semente é totalmente liberada de rastrear a cauda exponencial minúscula, bastando não exceder $(2\epsilon)^{1/n}$ nem ser negativa.
   - Perto de $x = 0$, a banda estreita-se automaticamente para compensar exatamente a derivada da cadeia de squarings:
     $$U_\epsilon(x) - e^{x/n} \approx \frac{\epsilon}{n e^{(1 - 1/n)x}}$$

---

## 3. Avaliação Técnica Especializada

### 3.1. Erro Absoluto vs. Erro Relativo na Softmax
Na Softmax normalizada $\hat{S}_i = \frac{\hat{E}_i}{\sum_j \hat{E}_j}$:
- A perturbação de primeira ordem demonstra que:
  $$\|\hat{S} - S\|_\infty \le 2 \|\hat{E} - e^x\|_\infty = 2\epsilon$$
- Tokens na cauda com logits baixos ($x_i \le -20$) contribuem com probabilidades $< 10^{-4}$ para a soma de atenção $\sum \hat{S}_i V_i$. Mesmo um erro relativo de $1000\%$ nessa região tem impacto nulo no produto escalar final.
- **Não-negatividade garantida:** Como a teoria impõe explicitamente $L_\epsilon(x) \ge 0$, os valores aproximados $\hat{E}_i$ são estritamente não-negativos, eliminando o risco de cancelamento destrutivo ou divisão por zero na Goldschmidt.
- **Conclusão:** O critério de **erro absoluto $\epsilon \le 10^{-3}$ é o ideal**. Tentar impor erro relativo exigiria graus desnecessariamente altos na cauda negativa.

### 3.2. Fronteira de Pareto e Oportunidades de Redução de Níveis CKKS
Dados empíricos da fronteira validada em `composed-frontier.json` (intervalo $[-32, 0]$, $\epsilon \le 10^{-3}$):

| Abordagem | Semente $d$ | Squarings $k$ | ct×ct Mults | ct×ct Depth | Níveis Estruturais | Erro Absoluto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chebyshev Direto (sem squaring)** | 13 | 0 | 7 | 4 | 5 | $6.42 \times 10^{-4}$ |
| **Composed Seed A (Min Níveis)** | **8** | **1** | **6** | **4** | **6** | **$7.81 \times 10^{-4}$** |
| **Composed Seed B (Min Multiplicações)**| **3** | **3** | **5** | **5** | **6** | **$7.81 \times 10^{-4}$** |
| **Perseus Atual (Desajustado)** | 8 | 2 | 8 | 5 | 7 | $\sim 10^{-3}$ (inflado em 0) |

- **Cenário 1 (Economia de Níveis):** Com $(d=8, k=1)$, economiza-se 1 nível multiplicativo e 2 multiplicações em relação ao Perseus atual.
- **Cenário 2 (Economia de Multiplicações):** Com $(d=3, k=3)$, o número de multiplicações ct-ct cai de 8 para 5 (redução de **$37.5\%$ no custo de tensores** da fase exponencial).

---

## 4. Plano de Ação para o Perseus

1. **Portar o Solver LP para o Calibrador:**
   - Adicionar o gerador `fit_composed_seed(domain, degree, k, eps)` em `python/perseus/calibrate/approximations/softmax.py` utilizando `scipy.optimize.linprog(..., method='highs')`.
2. **Atualizar o `gpt2.yaml`:**
   - Testar pares ótimos na calibração: $(d=3, k=3)$ vs $(d=8, k=1)$ vs $(d=4, k=2)$.
3. **Redução das Iterações de Goldschmidt:**
   - Devido à estabilidade superior dos denominadores $S_{\text{sum}}$, testar a redução de `gs_iters` de 8 para 6 (economizando 2 níveis multiplicativos profundos na divisão).

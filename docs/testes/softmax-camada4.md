# Teste do candidato 1 da softmax: camada 4 com δ2 = 4

Data: 2026-11-01. Origem: `docs/ideias/softmax.md`, item 1, e `subagent/softmax2/relatorio.md`.
**Parte de CPU feita aqui; a decisão depende da cauda dos dados e da ILP, que rodam na máquina do usuário.**

## A ideia

A camada 4 do recorde 4 é a única centrada com δ2 = 8: são 4 rodadas de normalização (Goldschmidt [6, 2, 3, 5]),
contra 2 nas outras centradas. O δ2 vem de `next_pow2(U*/τ)`, com τ = 4,6 e U\* = 18,569 o máximo dos dados. Como
U\*/4,6 = 4,037, a camada fica 0,9 % acima do limiar de δ2 = 4.

O upstream testou τ = 5,0, que só muda essa camada, e **recusou**: o KL de um modelo de ruído do refresh dobrou (README
do `2026-10-27_LeanChainDecode`). O teste aqui é outro: forçar δ2 = 4 e encolher a margem U/U\* (1,25 no recorde),
porque o ruído cresce com U.

## Resultados na CPU (`scripts/sm_l4_delta2.py`)

Gêmeo float64 do upstream (`softmax_ref`, `SM_CHEB_GS=1`), linhas sintéticas de 2 a 16 chaves, seis tipos de linha
(uniforme, agrupada, "real", plana, topo, fundo).

- **Reconstrução:** a camada 4 refeita com as fórmulas do `fit_centred` a partir da própria config é igual à do
  recorde (só `poly_coeffs` difere, em ~1e-15).
- **Ruído do 1º refresh:** ruído gaussiano absoluto na saída do exp, medido como rms|Δp|/σ. Nas camadas do recorde dá
  L0 50, L1 30, L2 66, L3 104, **L4 49**, L5 142, L6 56, L7 123, L8 30, L9 79, **L10 154 (pior centrada)**, L11 28.

| candidata | δ1 | δ2 | Goldschmidt | profundidade | max\|Δp\| | ruído | ÷ pior (L10) | Σ ruído² das 12 camadas | tolerância / U\* |
|---|---|---|---|---|---|---|---|---|---|
| recorde (margem 1,25) | 4 | 8 | [6, 2, 3, 5] | 33 | 4,2e-4 | 49 | 0,32 | ×1,00 | 1,56 |
| δ2 4, margem 1,25 (≈ o τ 5 recusado) | 8 | 4 | [8, 2, 5] | 30 | 4,2e-4 | 447 | 2,90 | ×3,17 | 1,40 |
| δ2 4, margem 1,10 | 8 | 4 | [6, 3, 5] | 29 | 5,5e-5 | 223 | 1,45 | ×1,52 | 1,25 |
| δ2 4, margem 1,05 | 8 | 4 | [7, 2, 5] | 29 | 4,2e-4 | 177 | 1,15 | ×1,32 | 1,20 |
| **δ2 4, margem 1,00** | 8 | 4 | [7, 2, 5] | **29** | 4,2e-4 | **140** | **0,91** | **×1,19** | **1,15** |

Como ler as colunas:
- **profundidade:** o `depth()` do script de ajuste, do exp até a última normalização;
- **Σ ruído²:** um proxy grosseiro do KL do modelo de ruído. Nele, o τ 5 recusado sai em ×3,2, e o upstream mediu
  KL ×2, então a ordem bate;
- **tolerância:** o maior score, em múltiplos de U\*, que uma chave pode ter sem a linha sair da softmax (erro > 0,01
  ou NaN), no pior kc.

Leitura:
- **Margem 1,00:** −4 níveis na cadeia da softmax da camada 4 e uma rodada a menos. O ruído fica abaixo do da pior
  camada que o recorde já aceita, e o proxy de KL sobe 19 %. O preço é a robustez: qualquer score acima de 1,15 U\*
  (21,4) quebra a linha. No recorde o limite é 1,56 U\* (29).
- **Margem 1,05:** compromisso, com tolerância 1,20 U\* e ruído 1,15× o da pior camada.
- **Ganho estimado:** −1 SparseB (≈ −8,4 ms) mais as ops da rodada que sai (1 quadrado, 1 soma por cabeça, 2 iterações
  de Goldschmidt, 1 produto), ≈ −10 ms (−0,6 %). Não basta sozinho para 1 %, mas soma ao EvalMod par (`docs/ATAP-ganhos.md`).

Configs prontas (recorde 4 com só a camada 4 trocada): `configs/softmax_l4/r4_l4d4_m{100,105,110,125}/configs.json`.

## Roteiro na máquina do usuário (do mais barato ao mais caro)

Os caminhos seguem o `HANDOFF-2026-10-30.md` (`work/perseus-r4` = recorde 4); ajuste se mudaram.

1. **Cauda dos dados (minutos, decide a margem).** Fora da calibração, quantas janelas passam de 1,15 / 1,20 / 1,25 U\*?

   ```bash
   python scripts/l4_tail.py --perseus work/perseus-r4 --calib --docs 20000 --c4 <c4-validation.00000-of-00008.json.gz>
   ```

   O `--calib` tem de dar max = 1,000 U\*, o que confirma que o script mede o mesmo que o ajuste. **Critério:** 0
   janelas acima da tolerância da margem escolhida, com folga. Se o máximo fora da calibração passar de ~1,1 U\*, a
   margem 1,00 está descartada.
2. **KL do modelo de ruído** (o critério que o upstream usou para recusar o τ 5). O script não está no snapshot do
   upstream que eu tenho. Se estiver com você, rode-o com as configs `m100` e `m105`. **Critério:** KL ≤ ~1,2× o do
   recorde.
3. **Captura + ILP**, só com a margem aprovada:

   ```bash
   cp -r configs/softmax_l4/r4_l4d4_m100 work/perseus-r4/python/perseus/configs/model/approximation/
   ```

   ```bash
   bash subagent/r4/cap0.sh graphs/r4_l4d4_m100 configs/model/approximation/r4_l4d4_m100/configs.json
   ```

   Depois, a captura fundida (`speedrun/make_decode_capture.sh`, `JOBS=1`, MemoryMax de 7 G, como em 2026-10-29) e o
   plano:

   ```bash
   MODE=base PLACER=ilp GR=graphs/r4_l4d4_m100m CONFIGS=configs/model/approximation/r4_l4d4_m100/configs.json bash subagent/r4/plan_n32a4.sh r4_l4d4_m100
   ```

   **Critério:** o bloco 4 com 1 refresh a menos (o recorde tem 10 nesse bloco), `compile_check` com 0 erros e
   `route_mix` com 0 fora da rota.
4. **KL float64** nas janelas de estresse e no hold-out (`scripts/kl_eval.py`, `subagent/audit/kl_windows.py` com
   `KLW_CFG`). Depois, o A/B na GPU junto com o pacote do EvalMod par.

## Reproduzir a parte de CPU

```bash
PYTHONPATH=<clone do perseus>/python python scripts/sm_l4_delta2.py configs/softmax_l4
```

Precisa de numpy e scipy e leva ~3 min. Sem `-I`: o script importa o upstream pelo `PYTHONPATH`.

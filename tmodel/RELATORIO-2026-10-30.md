# Modelo de tempo recalibrado (GPU 2026-10-30) e opções para o Decode-8

Tudo aqui rodou na CPU desta máquina (sem GPU, sem servidores). Números de tempo são do modelo, calibrado nos ms
medidos na RTX PRO 6000 WK (`gpu/results-2026-10-30-runpod/`); ms = ms/token na métrica do harness (posições 8-15).

## 1. O modelo

- **Unidades por op** (`units.py`): key switch por dígitos (`(β+2)(n+α)` NTT-limbs, β = ceil(n/13), α = 13 no n32a4),
  rotação hoisted, rescale composto, elementwise. Três constantes (ms por NTT-limb, por conversão de base, por limb
  elementwise) ajustadas aos estágios do pc-bench (raise, 4 CtS, EvalMod shipped e d27, 3 StC; K = 1, n32): erro
  ±6-15 % por estágio, EvalMod shipped e d27 ambos +8 % (as razões entre séries são confiáveis). `calib.py`.
- **Token** (`tokplan.py`, `predict.py`): censo de TODAS as ops dos 14 programas compilados (`Program.save` dá cada op e
  o nível de cada variável) + cada refresh pela rota, real/complexo, raise drop e série do EvalMod. SPRU (sem EvalMod,
  sem medida isolada) fica fixo em 8,0 ms (razão do route_table × 0,81).
- **Validação contra os A/B** (deltas sobre o recorde, mesma GPU; pares muito estáveis: ±0,6 ms em 6 pares):

  | plano | modelo | medido |
  |---|---|---|
  | LN(e3)+em9 `lnd_fin_e3_em9_cm` | +66,9 ms | +76,6 ± 0,6 |
  | LN(e3)+d27sb `lnd_fin_e3_d27sb_cm` | −3,4 ms | +5,9 ± 2,3 |

  Os dois erram pelo mesmo ~+9,5 ms. Duas explicações cabem igualmente: (a) a LN adiada custa ~10 ms que o censo de ops
  não vê; (b) a tabela d27 K=24 do SparseB custa ~+0,25 ms/refresh a mais que o modelo (34-38 SparseB nos dois planos).
  Em calibrações alternativas (sem o raise, só CtS+EvalMod, eps ×1,5) o Δd27sb fica em −3,3 a −4,4 e o Δem9 em +61 a +77.
- **Absoluto**: o modelo dá 1538 ms contra 1690 medidos; os ~150 ms que faltam cabem em ~6 µs por launch (25 mil
  launches/token) ou em erro de SPRU/IO. Só um perfil (nsys) separa isso.

## 2. Onde vão os 1690 ms do recorde 4 (modelo)

- Refreshes ≈ 1107 ms (65 %): 36 StC-first reais × 11,9 = 430; 6 StC-first complexos (CutMax, pares) × 15,5 = 93;
  40 SparseB-512 × 8,2 = 328; 32 SPRU × 8,0 (suposto) = 256.
- Ops ≈ 431 ms (26 %), sendo ~290 ms de key switch. Por etapa: LayerNorm 100 (ln_1 51 + ln_2 50; ~60 % disso é a
  1/√ num escalar a 29→17 limbs), softmax 74, GELU 42, CutMax 40, rescales "Realize" 36, up 22, out_proj 19,
  softmax_v 18, kv_push 18, qkv 16, tail 16, down 16, qkt 12.
- Resíduo ≈ 152 ms (9 %): launches/sincronização/IO, ou erro do modelo.

Valor de um refresh a menos (com os drops típicos): StC-first real ~11,9 ms, SparseB ~8,2, SPRU ~8 (não medido),
StC-first complexo ~15,5. O `subagent/r4/time_model.py` antigo dava 14,3 a todos os não-SPRU: superestimava o
SparseB em 75 %.

**Efeito de nível**: todas as ops 1 nível (2 limbs) mais rasas custam +47 ms; 1 nível mais fundas, −47 ms
(~24 ms por limb no token inteiro). É o que afundou o em9: os pousos 2 limbs mais altos encareceram as ops em
+52 ms, mais +46 ms do EvalMod d27 nas rotas densas, contra −31 ms de 6 SparseB a menos.

## 3. O que o modelo descarta

- **EvalMod de 9 níveis nas rotas densas**: no PS do Perseus (k,m do OpenFHE) toda série de 9 níveis tem mais relins
  que a enviada (grau 14 + 5 DA = 11): d13+5DA 12 (22,9 bits), d20+4DA 12 (29,1 bits), d27+4DA 13. Mesmo gastando o
  nível livre num drop (pouso igual ao do recorde), dense/StC-first fica +2 a +15 ms.
- **dnum**: 4 dígitos já é o ótimo; dnum 5 = +47 ms, dnum 6 = +94 ms (dnum 3 estoura log QP).
- **Economia de nível sem ajuste de pouso**: o smrel (−12 níveis) sobre LN+em9 dá +0,4 ms. A folga vira ops em mais
  limbs, não refresh a menos nem pouso mais fundo. O ILP minimiza a CONTAGEM de refreshes (objetivo "total") e o
  drop search só aceita drops que não mexem em nada a jusante.
- Os 6 StC-first complexos são do CutMax (pares complexos): não dá para virar rota real.
- Não há pares de refreshes independentes para batch K=2 (todos os 114 saem K=1).

## 4. Opções (ganho estimado pelo modelo; − = mais rápido)

Todos os planos abaixo compilam os 14 estágios com 0 erros, 114 refreshes (mesma contagem e mix do recorde) e 0
deliberados fora da rota. Planos em `work/perseus-r4/bootstrap_placements/`.

| opção | plano | modelo | o que muda | risco |
|---|---|---|---|---|
| A. SparseB na tabela d27 K24 + 1 raise drop | `sbd27d1_cm` | −9,4 (−9 a −11) | só runtime + drops do SparseB; níveis de TODAS as vars idênticos ao recorde | se o resíduo do d27sb for da tabela d27 (hipótese b), o ganho some |
| B. ILP com objetivo em ms (custos recalibrados) | `msobj_cm` | −7,6 | outro posicionamento, mesma contagem | pequeno, perto do erro do modelo |
| A+B | `msobj_sbd27d1_cm` | −17,3 (−1,0 %) | | o de A |
| C. 1/√ da LayerNorm a 1e-3 (recorde: 1e-5), mesmas faixas | `r4_ln3_cm` | −19,4 | graus 19-31 → 12-18; −23 níveis/token; KL gêmeo 1,922e-6 → 1,944e-6 (+1 %) | KL na GPU não medido |
| C+A | `r4_ln3_sbd27d1_cm` | **−28,8 (−1,7 %)** | | o de A e C |
| D. hoisting das rotações baby dos matvecs (7 rotações do mesmo ct, 49 grupos/token) | código | −10,6 (só ModUp); ~−19 com double hoisting | compilador + runtime; saídas idênticas byte a byte | engenharia de kernel |
| E. ILP com drops como variáveis + custo de ops por nível | pesquisa | teto −81 (todo refresh 1 nível mais fundo) | o drop search atual só aceita drops que não mexem em nada a jusante | desconhecido |

`r4_ln3_smrel_gelubs_cm` (C + smrel + GELU rasa; KL gêmeo 3,12e-6, ×1,6 por causa da GELU): −24,3 ms, só 4,9 ms além
de C; somando A ficaria ~−34, mas o KL sobe 60 %. `msdrop7_cm` (B com drops até 7 e pouso até 44): −7,6, idêntico a B. Os drops não estão presos pelo conjunto de
variantes nem pelo pouso máximo, e sim pelas regras do drop search (pinos a jusante). Rodou: `msdrop7` (B com drops até 7 e pouso até 44). Logs em `tmodel/run_*.log`; avaliar com
`evalplan.py` + `predict.py`.

Próximo passo sugerido (decisão sua; nada foi rodado em GPU): um A/B de `r4_ln3_sbd27d1_cm` contra o recorde. Ele mede
C+A e, com `sbd27d1_cm` no mesmo pod, separa as hipóteses (a)/(b) do resíduo de +9,5 ms. Um `nsys` de 1 token do
recorde explicaria os ~150 ms que o modelo não cobre e daria o custo real do SPRU.

## 5. Quanto nível falta para tirar um refresh por bloco (2026-10-30, manhã)

Plano hipotético, só do planner: receita e captura fundida do recorde, com os pousos k níveis (2k limbs) mais rasos do
que o runtime consegue, ILP nos blocos 0-12. Não compila; mede a folga necessária (`slack_sweep.sh`,
`slack_sweep2.sh`, logs `slack_sweep*.log`). Recorde: 101.

| folga | refreshes blocos 0-12 |
|---|---|
| k=1, todas as rotas | 100 |
| k=2, todas | **90** (−1 SparseB da softmax em 11 de 12 blocos) |
| k=3, todas | 89 (degrau: o próximo refresh pede muito mais) |
| k=2 só SparseB | 99 |
| k=2 só SPRU / só densos / só push K/V | 101 |
| k=2 SparseB + densos | **90** |
| k=2 SparseB + push K/V | 99 |
| LN adiada + smrel (captura `lnd_smrel_e3m`), EvalMod enviado, k=0 | 100 |
| LN adiada + smrel, k=1 em SparseB + densos | 95 |

Alvo: ~2 níveis a mais no caminho do q (refresh denso → LN → qkv → q·Kᵀ) E em cada trecho da softmax entre SparseB.
Vale ~11 SparseB ≈ 90 ms (~5 %), sem contar o efeito de nível. Fontes candidatas: LN adiada (−2 níveis no caminho
vetorial da LN, já implementada), smrel (~1 nível por camada) e a divisão da softmax (rodadas Goldschmidt 6+5
iterações; semente melhor para a rodada 0, ex. CGF ou minimax relativo do T1). Nada disso pode trazer o EvalMod d27
junto (caro demais nas rotas densas). LN adiada + smrel já cobrem parte: sozinhos tiram 1, e com +1 nível nos pousos
chegam a 95; para os 90 ainda faltam ~1-2 níveis por trecho de softmax/caminho do q além deles (o resíduo de +9,5 ms
da LN adiada, seção 1, entra na conta).

### Semente da rodada 0 da divisão da softmax (`cgf_seed.py`, gêmeo float64)

Denominador D de cada linha (camada, cabeça, posição) na softmax centrada do recorde 4, 48 janelas OWT de calibração +
96 de hold-out (221 mil linhas), margem ×3, alvo da rodada (0,2). Níveis da rodada 0 somados nas 12 camadas:

| semente | níveis/token | linhas fora no hold-out |
|---|---|---|
| recorde (linear, faixa estrutural [e^(−U/δ2)/16, 1,05]) | 78 | — |
| CGF (μ + σ²/2 de log z) | 79-80 | 0 |
| só a média | 84 | 1 |
| faixa de D ajustada aos dados | **64** | **0** |

A CGF morreu com a softmax centrada: a centragem na chave de referência já tirou a variação que μ e σ² explicavam
(no recorde 3 ela poupava 2-5 iterações por camada). Uma semente polinomial melhor também não ganha nível: o
Goldschmidt com escala de Chebyshev já é quase ótimo em profundidade para a faixa dada. O que sobra é estreitar a
faixa: −14 níveis/token (0 nas camadas 0-1, 1-2 nas outras), ao custo de trocar uma cota estrutural por uma calibrada
(linha fora = Goldschmidt diverge = NaN; exige verificação de cobertura muito maior). Config
`r4_ln3_smrel_db` (LN 1e-3 + smrel + faixa de dados): KL gêmeo 1,948e-6 (recorde 1,922e-6).

Faixa de dados DESCARTADA (decisão do usuário): o lado folgado da faixa do recorde é o TETO (1,05 garantido contra
D máximo 0,01-0,07 nos dados), e é acima do teto que o Goldschmidt diverge. Só o piso (lado seguro: abaixo dele o
resíduo T_n(·)/T_n(z0) fica em (0, 1), a recíproca sai menor mas finita) piora: 88 níveis. Afrouxar o alvo final
(1e-3 → 1e-2) poupa 0-4 níveis/token. A divisão da softmax do recorde 4 está no piso; o Goldschmidt escalado por
Chebyshev JÁ é o minimax relativo de 1/x (resíduo T_{2^k}(·)/T_{2^k}(z0) = o polinômio ótimo com q(0) = 1, em
profundidade log2 do grau). A rodada com a faixa de dados (`run_lndcfg.sh`) morreu na captura fundida (exit 124) e
não foi refeita. O que limita a softmax é a regra do δ2 (exponencial a ≤ e^4,6 da soma por causa do ruído do refresh
SparseB): refresh mais preciso tiraria uma rodada de normalização (~4-5 níveis por camada). Não estimado.

### LayerNorm com T1 (`ln_t1.py`)

Por sítio, no alvo 1e-3 e na faixa do recorde: polinômio direto (o ajuste repesado do recorde já é minimax relativo,
ou seja, o T1 de um polinômio só) contra semente + passos de Newton escalado minimax (refinador homogêneo do T1,
p = −1/2, forma fechada por equioscilação). O composto NUNCA ganha nessas faixas (razão 12-35; Newton custa 2
níveis/passo). Níveis da 1/√ por token: recorde 187, alvo 1e-3 164, alvo 1e-3 + dobra 139.

Dobra (`LN_CHEB_FOLD=1`, `python/perseus/impl/norm.py`, worktree `ln-defer`, SEM commit, só o `norm()` do decode sem
LN adiada): o ×1/d e o +ε da variância entram no intervalo do Chebyshev (a afim lê a soma da classe), álgebra exata.
Captura da posição 0 (`graphs/r4_ln3f`): KL igual (9,946e-7), −23 mults e −23 adds na LN, refreshes reativos da
captura na LN 65 → 36. Para servir precisa virar campo do pc-config (como `model.ln_defer`).
Plano completo `r4_ln3f_cm` (`run_r4_ln3f.log`): 114, 0 erros, 0 deliberados fora da rota; modelo −18,7 ms (sem dobra
−19,4). Com o SparseB d27 + drop: `r4_ln3f_sbd27d1_cm` −28,1. O drop search usou o nível: SPRU da LN2 com drop 3 → 5,
pouso 20 → 24, soma dos drops SPRU 131 → 167. O modelo não vê ganho porque cobra o SPRU fixo (8 ms, sem medida isolada);
um SPRU com drop maior é mais barato (o produto das chaves roda d níveis abaixo), então o ganho real da dobra fica entre
~0 e ~10 ms conforme essa sensibilidade, que só uma medida na GPU dá.

## 6. A/B na GPU (2026-10-30, tarde): o modelo confirmou

Runpod, RTX PRO 6000 Max-Q (300 W, Community, US$ 1,64/h; WK/Server sem estoque), pod `ovb84iu9nq8x52` terminado,
~55 min ≈ US$ 1,50. Fora do harness (`NOSANDBOX=1`, `harness: null`: não reportável). Entrada = branch `tmodel-gpu`
34e8db7 (bundle `gpu/perseus-tmodel-entry.bundle`, script `gpu/run_gpu2.sh` + `gpu/go_tmodel.sh`). Resultados em
`gpu/results-2026-10-30b-runpod/`.

| plano (3 pares alternados) | entrada | recorde 4 | razão | KL | score.py | modelo (WK) |
|---|---|---|---|---|---|---|
| `r4_ln3_sbd27d1_cm` | 2,324 s | 2,369 s | **×1,019** (p 0,0003) | 4,5e-6 | NEW RECORD | −1,7 % |
| `sbd27d1_cm` | 2,357 s | 2,373 s | ×1,007 (p 0,007) | 7,6e-6 | abaixo da margem | −0,55 % |

- O ganho se divide em ~0,7 % (SparseB d27 + drop) e ~1,2 % (LN 1e-3), perto do modelo. A Max-Q é ~1,4× mais lenta
  que a WK; em proporção, o modelo acertou os dois.
- O SparseB sozinho rendeu o previsto ⇒ a tabela d27 K24 NÃO é mais cara que o modelo supõe; o resíduo de +9,5 ms de
  ontem era da LN adiada (hipótese a).
- Próximo para virar PR: medir no harness oficial (VM/bare metal com bwrap + systemd), na placa de referência.

## 7. Arquivos

`tmodel/`: `units.py`, `calib.py` (+ `calib.json`), `tokplan.py`, `predict.py`, `steps.py` (ms por etapa),
`stepops.py` (ops de uma etapa), `ksdist.py` (key switch por limbs, efeito de nível), `hoist.py`, `census.py`,
`saveprogs.py`, `emcost_main.rs` (censo do EvalMod via pc-shadow; crate de rascunho no scratchpad), `em/` (listas de
ops do EvalMod por série e topo), `ln_refit.py`, `run_*.sh` / `run_*.log` (planos da noite), `plan_n32a4_add.sh`
(cópia do script de plano com `RECIPE_ADD=`). Uso: `python -I predict.py PROGS_DIR record <plano>` com os programas
salvos por `saveprogs.py` (`Program.save`).

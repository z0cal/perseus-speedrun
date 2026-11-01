# Auditoria: plano Decode-8 de 98 bootstraps/token (`fin_dfr13_m_cm`)

Auditor independente, 2026-10-27. Tudo o que está abaixo foi lido no código ou medido na CPU nesta sessão. Os scripts
da auditoria estão em `subagent/audit/` (nada fora dessa pasta foi alterado):

- `compile_check.py`: compila um plano com o `_pc` do build;
- `band_tail.py`: folga da faixa da rodada 0 em janelas novas;
- `kl_windows.py`: KL float64 por janela, com a config final e com a do recorde;
- `emops/`: mix de operações do EvalMod;
- `old_*.json` / `new_*.json`: tabelas do EvalMod extraídas dos dois commits.

## Veredito em uma linha

**NO-GO para alugar GPU agora.** Achei dois bloqueadores, os dois verificáveis e corrigíveis na CPU:

1. O plano **não compila** no próprio build em9. O servidor morreria no `prepare()`.
2. A faixa recalibrada da rodada 0 do Goldschmidt **diverge (KL = inf)** em cerca de 0,1-0,2 % das janelas de texto
   comum. Isso basta para reprovar o teste de KL do conjunto oculto.

Os dois têm conserto. Depois do conserto, a contagem de bootstraps tende a subir um pouco.

---

## Riscos, do mais grave ao menos grave

### R1. ALTO, certeza: o plano em9 não compila (as tabelas de pouso do runtime ainda são as do recorde)

- **Evidência (medida).** `compile_check.py` chama `_pc.Program.compile(graph, plan)`, o mesmo caminho do
  `PlannedRun.prepare` em `impl/driver.py:653`.

  | build | plano | grafo | erros de checagem |
  |---|---|---|---|
  | em9 | recorde `gpt2_decode_python_n32_kc16_cm` | `kc16m` | **0** (o harness é fiel) |
  | em9 | `fin_dfr13_m_cm` | `fin_dfr13_m` | **31.551** |
  | em9 | `plans/gpt2_decode_em9_cm` | `kc16m` | centenas por bloco |

  O primeiro erro, no bloco 0: `refresh v_445: refresh input level 50 > 48 (BTS_MAX_INPUT_LEVEL): returns garbage`.
  Depois dele vêm em cascata os `rule rescale: top limb 1 has no 2 primes left`.
- **Causa.** O commit `67717aa` mudou `EVALMOD_LEVELS` e as tabelas, mas não mexeu no que o runtime acha que é o pouso.
  Ficaram com os valores do recorde:
  - `crates/pc-program/src/compile/mod.rs:238-240`: `runtime_shadow` fixa os pousos do FIDESlib em
    `(0, l(28)), (512, l(24)), ...`;
  - `crates/pc-shadow/src/route.rs:89-104`: `BtsConfig::n32_shipped` (28 / 26 / 34);
  - `python/perseus/impl/fakecap.py:144`: o gêmeo de CPU usa `0:32,1:14,256:22,512:22` por padrão. A captura
    `fin_dfr13` ainda registra o refresh deliberado do push K/V em 26 grau 2, e só o remap `26:24` da receita esconde
    isso do planner.

  O planner foi alimentado com os pousos novos pela receita (BTS_LEVEL=26, 512:24, fallback 32). O compilador e o gêmeo
  de CPU do servidor continuam nos antigos. Na GPU, o pouso real vem do `out_meta` do programa EvalMod
  (`bts/dense.rs:511`, `sparse.rs:221`, `stc_first.rs:844`), ou seja, 2 limbs mais raso.
- **Consequência.**
  - Hoje: com o plano em9, o servidor falha no setup.
  - Com o plano do recorde no build em9: compila, mas a GPU entrega ciphertexts 2 limbs fora do que o programa
    compilado espera.
- **Sintomas colaterais.**
  - O teste `route::tests::multi_stage_landings` falha (rodei: `left 22, right 24`).
  - `crates/pc-core/tests/oracle_cheb.rs` falha (já previsto no README).
  - `examples/em_depth.rs:24-25` ainda passa DA = 5 para as tabelas padrão, então sem argumentos imprime "20 levels"
    para a tabela nova. Engana quem rodar.
- **Para confirmar o conserto (CPU, minutos).**
  1. Atualizar os três lugares, de preferência derivando de `EVALMOD_LEVELS` em vez de repetir números.
  2. Recompilar o `_pc`.
  3. Recapturar e rodar `compile_check.py` no plano final: tem que dar **0 erros nos 14 estágios** (blocos 0-12 e o
     CutMax 13).
  4. Repetir com o plano do recorde no build novo: tem que FALHAR, o que prova que o compilador agora vê os pousos
     novos.

  Se o gêmeo fakecap mudar o pouso do push K/V, o remap `26:24` deixa de ser necessário e precisa sair. Senão ele
  vira `24 → 22`, otimista demais.

### R2. ALTO, provável: a faixa da rodada 0 (×4 embaixo, ×6 em cima) não cobre a cauda; o Goldschmidt diverge

- **Evidência (medida).** `band_tail.py` calcula, para cada camada, o divisor da rodada 0
  `d = mean_k exp((S_k - mid) / 2^l2)`. É a mesma fórmula do `headroom_split.py`, com scores float64 exatos.

  Janelas fora da faixa nova:

  | conjunto | janelas fora |
  |---|---|
  | 1.763 inícios de documento do pool OWT | **3 (0,17 %)** |
  | 2.000 offsets aleatórios abaixo de 1.000.000 (mesma distribuição da calibração) | **2 (0,10 %)** |

  Piores folgas de cima (< 1 significa fora):

  | camada | l2 | início de documento | aleatório |
  |---|---|---|---|
  | L10 | 0 | **0,08** (divisor 1,6e4, topo 1.230) | 0,54 |
  | L6 | 1 | 0,89 | — |
  | L7 | 0 | — | 0,78 |
  | L8 | 0 | 1,23 | — |

  Nas mesmas janelas, a faixa do recorde tem folga mínima de **1,98×** em todas as camadas.
- **Confirmação no modelo de referência.** `kl_windows.py` usa o `forward_approx` do `kl_eval.py`, isto é, o
  `ref.softmax_ref` do Perseus com `SM_CHEB_GS=1`. As 5 janelas sinalizadas (offsets 1094059, 1149233, 1678699,
  257792 e 853437) dão **KL = inf com a config final** e 2e-7 a 3e-6 com a do recorde.
- **Por que acontece.**
  1. O split escolheu `l2 = 0` nas camadas 5, 7, 8, 9 e 10. Com isso o divisor vira `exp(S - mid)` sem compressão, e
     cada +0,7 no score dobra o divisor.
  2. A calibração usa só 256 janelas.
  3. A "folga ≥ 3,4×" foi medida em 80 janelas, e 32 delas são o conjunto dev. O README dava isso como evidência de
     segurança.
  4. O recorde fez diferente:
     - `l2 ≥ 1` em todas as camadas;
     - topo garantido estruturalmente ("its top at least 1.05× the head sum of a query at the range top", nas notas
       de `gpt2_base_n32_kc16_cm`);
     - verificação em 16.384 prompts de validação do C4.
  5. A docstring de `scripts/sm_split.py:4` diz "capped by the record's top", mas a linha 73 só multiplica a faixa.
     Não há cap nenhum.
- **Impacto.** Uma janela que diverge na CKKS dá lixo ou valores enormes, então o KL daquele prompt explode. O teste
  do harness compara a **média** do KL com 1e-3 a p < 0,01. Com 0,1-0,2 % por prompt, um conjunto oculto de 50-100
  prompts tem cerca de 5-18 % de chance de pegar pelo menos um. O resultado seria rejeição ou um resultado oficial
  ruim.
- **Conserto e checagem (CPU).**
  - Recalibrar as faixas (ou proibir `l2 = 0`) com:
    - ≥ 16k janelas, inícios de documento incluídos;
    - topo estrutural como o do recorde.
  - Rodar `band_tail.py` em ≥ 16k janelas que nunca entraram em escolha nenhuma. Critério: 0 janelas fora, folga
    mínima ≥ 1,5-2×.
  - Rodar `kl_windows.py` nas piores janelas.
  - Depois, re-ILP. Espere perder parte do ganho da softmax: o "×4/×4 = 94" e o "×4/×6" deixam de ser comparáveis.

### R3. MÉDIO, provável: a contagem de bootstraps superestima o ganho de tempo

- **Mix de rotas (contado nas placements).** Blocos 0-12; o estágio 13 do CutMax tem 12 dense nos dois.

  | plano | dense | SparseB-512 | SPRU |
  |---|---|---|---|
  | recorde | 36 | 47 | 25 |
  | final | 36 | **25** | 25 |

  Todos os 22 bootstraps economizados são SparseB-512, os da softmax. Nenhum dense saiu.
- **Estimativa de tempo** com o `bts_ms` de `plan/data/route_table_n32.json` (stcf 17,03, 512 17,39, SPRU 9,98 ms):

  | | bootstraps por token |
  |---|---|
  | recorde | ≈ 1.885 ms |
  | final | ≈ 1.502 ms |
  | diferença | **≈ −383 ms (≈ −16 % de 2.325 ms/token)**, antes dos efeitos abaixo |
- **Efeitos contra o ganho (não medidos).**
  1. **EvalMod dense mais caro.** Pelo `emops`, que usa o `build_evalmod` real:
     - a tabela nova tem 100 ops brutas (73 fundidas), contra 80 (52) da antiga;
     - 13 relinearizações contra 11;
     - soma de limbs nos produtos 514 contra 428 (**+20 %**).

     São 48 dense por token, incluindo o CutMax. Se o EvalMod for cerca de 30 % do bootstrap dense, o custo sobe uns
     +6 %, ≈ +50-70 ms/token. No SparseB a tabela tem o mesmo custo: 100 ops contra 99.
  2. **Menos raise drops nos SparseB.** O drop médio dos 512 cai de 3,1 (recorde) para 1,5 (final). Drop maior é
     bootstrap mais barato, então cada um dos 25 restantes custa mais. A tabela não traz latência por drop.
  3. **Pousos 2 limbs mais ricos.** As operações depois do pouso rodam com cerca de 2 limbs a mais (≈ +6 % por op fora
     de bootstrap).
- **Expectativa realista.** Ganho de **≈ 10-15 %**, não 18,3 %. Ainda acima da margem de 1 %, mas só a GPU decide.
  Observação: a ILP minimiza contagem (`ilp_objective="total"`), igual ao recorde, então a comparação de contagem é
  coerente. O `PLAN_ILP_OBJECTIVE=ms` (item 5 do subagente) ficaria para depois de medir.

### R4. MÉDIO, provável: margem escolhida olhando o conjunto dev (regra "do not tune anything to the dev prompts")

- **Evidência.**
  - `scripts/headroom_split.py:22` e `headroom.py:20` medem a folga em `windows("dev", 32) + windows("owt", 48)`.
  - O README rejeita "×4/×4 (2,3×, arriscado)" e escolhe "×4/×6 (≥ 3,4×)" com base nessa medida.
  - `scripts/final.py` também mede a faixa contra o dev.

  É uma escolha de parâmetro informada pelo dev. Foi na direção conservadora, mas é escolha.
- **Conserto.** Refazer a decisão em dados que não sejam o dev: inícios de documento do OWT, ou validação do C4 em
  shard nunca usado. O R2 já obriga a isso. Usar o dev só para o número final e declarar em Disclosures, como o
  recorde fez ("dev prompts 0, 2 and 3 were looked at in diagnostics…").

### R5. MÉDIO, provável: o Prefill-128 quebra (EvalMod é constante de compilação, não chave de runtime)

- **Evidência.**
  - `EVALMOD_CHEB`, `*_DA_ITS` e `EVALMOD_LEVELS` são `const`. Valem para todas as rotas e todos os planos.
  - O plano do prefill (`gpt2_prefill128_pfbhrd_cm`) foi planejado com os pousos antigos.
  - O prefill usa SparseB com K = 16 (`FIDESLIB_BTS_SPARSE_B_K=16`, BabyStepPrefill), que agora pega a tabela d27 com
    4 DA.
  - Depois do R1 consertado, o prefill falha a compilação. Antes do conserto, roda desalinhado na GPU.
- **Impacto.** Os mantenedores medem o commit. Um PR que quebra o outro track, ou que troca a aproximação do prefill
  sem declarar, tende a ser recusado. "Falta 2/3" do README já aponta isso.
- **Conserto.** Tabela atrás de um switch por plano (linha `runtime=`), com o padrão igual ao do recorde. Isso também
  mantém o `oracle_cheb.rs` passando. Ou então replanejar o prefill e declarar.

### R6. MÉDIO, possível: magnitudes nas camadas com `l2 = 0` e o CF dos refreshes da softmax

- **Evidência.** O `fin_dfr13_merge.log` mostra que a captura fundida (4 linhas × 16 posições) aumentou muito alguns
  nós da softmax em relação à posição 0:
  - bloco 5: `v_28 x1004`;
  - bloco 10: `v_28 x168`;
  - bloco 7: `v_379 x199`.

  O CF de cada refresh sai do máximo dessas 4 linhas vezes `PLAN_MAG_SAFETY=2`. Com `l2 = 0`, uma janela da cauda
  ultrapassa ×2 com +0,7 no score: entrada do bootstrap fora da faixa do CF, a senoide dobra e sai lixo. É o mesmo
  mecanismo do R2, agora nos bootstraps. O recorde, com `l2 ≥ 1`, comprime essas faixas.
- **Checagem (CPU).** Depois do conserto do R2:
  - comparar, nó a nó, o máximo do `capture_mags` em ≥ 64 linhas com o envelope do CF do plano, que é o
    `good_bounds` de `bts_accuracy_n32.json` aplicado ao CF de cada site;
  - ou, no mínimo, rodar a captura fundida com `ROWS` de 16 a 32 linhas e ver se o plano muda.

### R7. BAIXO-MÉDIO: `bts_accuracy_n32.json` / `good_bounds` medidos com o EvalMod antigo

- **Precisão de aproximação.** O modelo float64 (`em_fit.py`) bate com o pipeline:
  - c0 armazenado sem a metade, e a metade aplicada em `add_scalar(c0/2)` no `pc-shadow/src/evalmod.rs`;
  - constantes dos ângulos duplos `-(2π)^(-2^(j-r))` (`evalmod.rs:793-801`), iguais à recorrência `s_{j+1} = s_j²`;
  - entrada `u/K` (`sparse_b.rs:248`).

  Reproduzi as tabelas:

  | tabela | antes | agora |
  |---|---|---|
  | dense | 27,4 bits | 43,5 bits |
  | K24 | 35,2 bits | 28,2-29,1 bits |
- **Comparação com o piso medido.** No `bts_accuracy` (rota 512, `rel_err·amp/2^cf` mínimo) o erro absoluto mínimo é
  ≈ 2^-23. Os 2^-29 da tabela K24 nova ficam 64× abaixo disso (≈ +1,5 % no pior ponto). O dense não muda, porque 2^-27
  já estava abaixo do piso de ruído.
- **Ruído.** A amplificação dos ângulos duplos cai de ≈ 4^5·(2π)^-0,97 ≈ 172 para ≈ 46, então o ruído das rescalas no
  polinômio pesa menos. A massa de coeficientes (2,24 contra 1,87) é irrelevante nessa escala, porque `|T_k| ≤ 1`.
- **Borda do domínio.** Em |I| = K a tabela K24 nova cai para ≈ 24 bits (a antiga dava ≈ 40). Só importa num evento
  raro de |I| = 24.
- **Conclusão.** As faixas de CF provavelmente continuam válidas, talvez até um pouco melhores, mas é suposição até
  medir. `pc-bench bts --all` na GPU tem que regenerar as curvas, e se elas mudarem, re-ILP.
- **Nit.** O dense "grau 27" é na verdade grau 26: o último coeficiente é `-0.0`, e `degree()` o ignora. O PS sai
  com k = 4, m = 3, o mesmo caminho genérico que o SparseB já roda na GPU. Corrigir a descrição.

### R8. BAIXO-MÉDIO: evidência de KL só em float64 e posição 0

- **O que foi medido.** O `kl_eval.py` não tem ruído CKKS, erro de bootstrap, nem `defer_norm`. Em float64 o
  `defer_norm` é idêntico, então não está testado ali.
- **Medi em 160 inícios de documento OWT** (sem as janelas do R2):

  | config | KL médio | KL máximo |
  |---|---|---|
  | final | 3,6e-6 | 1,3e-5 |
  | recorde | 1,8e-6 | 6,2e-6 |

  O recorde cifrado no dev deu 3,9e-6, contra ≈ 2e-6 em float64. Então a margem até 1e-3 continua enorme, desde que
  não haja evento catastrófico (R2/R6).
- **O gêmeo de CPU com `defer_norm`** só foi conferido contra o oráculo na posição 0 (`[decode] PASS`, kc = 1). A
  captura fundida das posições 0-15 não compara com o oráculo.
- **O que pioraria muito o KL cifrado.**
  - Divergência da faixa (R2).
  - Estouro de CF (R6).
  - Pousos errados (R1).
  - Desalinhamento do 1/s após o rotsum para kc > 1. É improvável: 1/s é constante nos t slots de cada cabeça e
    comuta com o rotsum e a máscara. Mas não está testado.
- **Checagem (CPU).** Rodar o gêmeo numpy em todas as 16 posições com o oráculo e conferir KL e top1 por posição,
  com e sem `defer_norm` (o `run_decode --fake --tokens 16`, se suportar).

### R9. BAIXO: estágio 13 (CutMax) reaproveita a captura do recorde

- **O que é reaproveitado.** O `make_cutmax_plan.sh` usa `graphs/gpt2_decode_python_n32_kc16_cm`, capturado com o
  `_pc` e as aproximações do recorde.
- **Por que deve servir.**
  - O CutMax só lê os logits, e as magnitudes deles quase não mudam (o KL é minúsculo).
  - O pouso deliberado passa pelo mesmo remap.
- **O que fica no limite.** A saída do estágio sai em `exit 48:2`, exatamente na guarda `BTS_MAX_INPUT_LEVEL = 48`.
  A armadilha da camada 11 no README mostra como isso é sensível.
- **Checagem.** O R1 consertado precisa compilar o bloco 13 sem erro. Na GPU, argmax ≥ 95 % (o recorde fez 96/96).

### R10. BAIXO: legibilidade e manutenção

- `defer_norm` acrescenta um ramo pequeno e claro.
- O `tap("attn.probs")` some quando ele está ligado (`attention.py`, função `mha`). Nenhum consumidor no repo, mas
  ferramentas de depuração perdem o tap.
- As configs por camada seguem o padrão do recorde.
- Revisores vão estranhar duas coisas, que devem ser documentadas:
  - a troca do EvalMod padrão sem switch (R5);
  - o remap `26:24` na receita.

---

## O que está sólido

- **Segurança.**
  - K = 16 / K = 24, segredo, chaves e cliente não mudam.
  - O cliente pinado (`perseus_core_client.py`) não lê plano nem config; só o nível de entrada.
  - Nada vaza para fora do relógio.
  - Execução oblívia: as configs por camada e o `defer_norm` são estáticos. `sm_periodic` depende de `k_count`, que
    é público.
- **Profundidade do EvalMod.** 9 níveis (18 limbs contra 20), confirmado com o `build_evalmod` real (`em_depth` e
  `emops`).
- **Modelo de erro do EvalMod.** Fiel ao pipeline (convenção de c0, constantes dos DA, escala por K). A queda do K24
  para 28-29 bits fica abaixo do piso de ruído medido.
- **Contagem.**
  - A ILP tem o mesmo objetivo do recorde.
  - O plano fundido (86) é consistente com o da posição 0 (87).
  - O estágio 13 continua com 12.
- **Fora das janelas patológicas**, o KL float64 da config final fica em ≈ 2× o do recorde, ainda 250× abaixo da
  barra.
- **A lógica do `defer_norm`** está correta para `SM_PERIODIC = 1` (Decode-8 ≤ 16 chaves). Nos outros ramos, a
  multiplicação acontece antes do produto com V, o que é seguro.

## Recomendação

**NO-GO para gastar com GPU agora.** Ordem sugerida, toda na CPU:

1. **R1.** Pousos derivados de `EVALMOD_LEVELS` em `compile/mod.rs`, `route.rs` e `fakecap`. Rebuild, recaptura e
   `compile_check.py` com 0 erros em 14 estágios.
2. **R2/R4/R6.** Faixas da rodada 0 com topo estrutural e calibração ≥ 16k (inícios de documento incluídos, dev
   fora). Depois `band_tail.py` e `kl_windows.py` em held-out novo, com 0 divergências. Depois re-ILP e nova
   contagem.
3. **R5.** EvalMod atrás de um switch por plano, com o padrão do recorde. Conferir que o prefill do recorde ainda
   compila.

**Na rodada de GPU, medir no mínimo:**

1. `pc-bench bts --all`, build em9 contra o recorde:
   - latência por rota: dense stcf, dense shipped, SparseB-512 em cada raise drop usado, SPRU;
   - erro relativo × amplitude por CF, para regenerar o `bts_accuracy_n32.json`.
2. Harness Decode-8 intercalando recorde e entrada (≥ 6 pares, a mesma GPU ociosa, de preferência a RTX PRO 6000)
   e `score.py`: tempo e teste de KL.
3. KL cifrado e argmax:
   - dev 0-15;
   - um held-out limpo (C4 validation 32-79, como o recorde);
   - teste de estresse com as 5 janelas OWT do R2 (1094059, 1149233, 1678699, 257792, 853437): nenhuma pode dar
     NaN ou KL > 1e-3.
4. Perfil por estágio: tempo de bootstrap por rota e tempo fora de bootstrap, para validar o R3. Em especial o custo
   do EvalMod dense novo e o efeito dos pousos mais ricos nas operações.
5. Prefill-128 no mesmo commit: tem que rodar (switch ou replanejamento), com tempo e KL ao lado do recorde.
6. Pico de memória da GPU (o recorde: 56,5 GiB numa H100).

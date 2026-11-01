# Perseus speedrun — Decode-8

Desafio: https://github.com/gladia-research-group/perseus/tree/speedrun. Recorde vigente: **Decode-8 recorde 4**
(`origin/speedrun` = d97a6f7, 1,798 s/token).

## Estado em 2026-11-01 (leia isto primeiro)

- **Recorde vigente:** recorde 4, 114 refreshes/token, cadeia n32a4, softmax centrada. Regra 5: a entrada precisa ser
  ≥ 1 % mais rápida no harness oficial, com KL < 1e-3.
- **Melhor candidato medido:** `r4_ln3_sbd27d1_cm`, que junta a 1/√ da LayerNorm a 1e-3 com o SparseB na tabela d27
  K24 e +1 raise drop. Mantém os mesmos 114 refreshes e os pousos do recorde. Deu **×1,019** contra o recorde 4 na
  mesma GPU (RTX PRO 6000 Max-Q, 3 pares alternados, p = 0,0003, KL 4,5e-6). Foi medido **fora do harness**
  (`NOSANDBOX=1`), então não é reportável. Detalhes em `tmodel/RELATORIO-2026-10-30.md`, seção 6.
- **Falta para PR:** rodar no harness oficial numa VM ou bare metal. A Runpod é container, e o `sandbox.sh` não sobe lá.
- **Linhas fechadas (não repetir):**
  - EvalMod d27 nas rotas densas: ×0,956. O EvalMod fica 27 % mais caro e o pouso mais alto encarece as ops.
  - d27 só no SparseB, sem drop: empate.
  - smrel, faixas da softmax e LN adiada sobre o recorde 4: 0 refreshes a menos.
  - Semente CGF, T1/Newton na LN, f∘arcsin (RBOOT), GELU par (−30 produtos/token, ~0,2 %), semente do exp por
    bandas (recai no smrel).
- **Próximas ideias, por ganho estimado:**
  - EvalMod par z11 nas rotas densas com +1 raise drop, mais o par z13 no SparseB: −25 a −45 ms/token, sem replanejar
    (`ATAP-ganhos.md`);
  - hoisting das rotações baby (opção D do `tmodel`): −10 a −19 ms;
  - ILP com objetivo em ms (opção B): −7,6 ms.

### Documentos

| arquivo | conteúdo |
|---|---|
| `HANDOFF-2026-10-30.md` | passagem mais recente: GPU, ferramentas, restrições da máquina |
| `tmodel/RELATORIO-2026-10-30.md` | modelo de tempo calibrado na GPU, opções A–E, A/B do ×1,019 |
| `ATAP-ganhos.md` | ideias do Trefethen (ATAP) com contagens e prioridades |
| `IDEIA-evalmod-par.md` | EvalMod par: matemática, tabelas de precisão e custo, como implementar |
| `IDEIA-gelu-par.md` | GELU par: contagem real (ganho pequeno) |
| `IDEIA-softmax-ponderada.md` | semente do exp por bandas da cadeia de quadraturas: fechada (recai no smrel) |
| `subagent/audit/auditoria.md`, `subagent/sanity*/relatorio.md`, `subagent/r4/relatorio.md` | auditorias e testes de sanidade |
| `subagent/ganhos-rapidos.md` | smrel, gelubs, normalização adiada (recorde 3) |

### Código e artefatos

| arquivo | conteúdo |
|---|---|
| `gpu/perseus-tmodel-entry.bundle` | branch `tmodel-gpu` 34e8db7, a entrada do A/B ×1,019 (sobre o recorde 4) |
| `perseus-lndefer-entry.bundle` | branch `ln-defer` 83b8d55 (recorde 4 + em9/d27sb + LN adiada) |
| `perseus-evalmod-switch.bundle`, `0001-EvalMod-degree-27-tables-behind-FIDESLIB_BTS_EVALMOD.patch` | switch do EvalMod sobre o recorde 3 (d8a7643) |
| `perseus-evalmod-d27.bundle`, `0001-EvalMod-degree-27-series-…patch` | primeira versão do d27 (recorde 3) e o branch `defer-norm` (0b32814) |
| `fits/` | coeficientes do EvalMod: `full_d27_r4.json` (dense em9), `k24_full_d27_r4.json` (SparseB), `even_*.json` (par) |
| `plans/` | planos do recorde 3: `fin_rc3b_m_cm` (110/token, o válido), `gpt2_decode_em9{,_cm}` (fonte da receita em9 de `final_plan.sh` / `plan_count.sh`) |
| `scripts/`, `subagent/`, `tmodel/`, `gpu/` | ferramentas; ver `HANDOFF-2026-10-30.md` e os relatórios |

`work/` (clones do Perseus, capturas, ILPs) não é versionado: os caminhos `work/...` citados abaixo só existem na
máquina local. Para reproduzir os planos sem GPU: `sudo pacman -S cuda`; no clone, `uv venv .venv --python 3.12`,
`uv pip install maturin numpy scipy==1.18.1 networkx mpmath omegaconf` e `CUDA_HOME=/opt/cuda maturin develop --release`.

## Histórico do recorde 3 (2026-10-26 a 10-28), resumido

As seções originais destes dias tinham contagens que depois se mostraram erradas: 113 (em9), 106 e 98 refreshes/token
saíram todas com o bug do remap duplo (seção seguinte). O texto completo está no histórico do git. O que continua
valendo:

- **EvalMod de 9 níveis.** O EvalMod do recorde 3 (`crates/pc-core/src/bts_setup.rs`) era grau 14 + 5 ângulos duplos
  (10 níveis); o grau 14 fica 1 acima do limiar do Paterson-Stockmeyer. O refit minimax foi feito no intervalo inteiro
  (|u| < 16; os `fits/full_*` usam eps de ajuste 0,5) e ponderado pela amplificação dos ângulos duplos (`scripts/em_fit.py`):

  | EvalMod | níveis | precisão |
  |---|---|---|
  | recorde: grau 14 + 5 DA (K=16) | 10 | 27,4 bits |
  | grau 27 + 4 DA (K=16), em9 | 9 | 43,5 bits |
  | grau 27 + 3 DA (K=16), em8 | 8 | 23,7 bits |
  | SparseB recorde: grau 20 + 5 DA (K=24) | 10 | 35,2 bits |
  | SparseB grau 27 + 4 DA (K=24), em9 | 9 | 28,2 bits |

  A rota SPRU (1 slot, "decryption in the exponent") não usa EvalMod. Na GPU o d27 saiu 27 % mais caro por refresh
  (2026-10-30).
- **Bootstrap funcional (RBOOT), ≈ 0 níveis.** 33 de 101 refreshes antecedem uma não-linearidade numa rota em que o
  EvalMod age na mensagem: 24 SPRU → 1/√ da LayerNorm e 9 StC-first → GELU. Os 40 da softmax são SparseB, na ordem
  clássica, e não servem. Mesmo assim os ângulos duplos já têm profundidade mínima, e f∘arcsin custa o mesmo grau que f
  para funções suaves (1/√ grau 23, GELU grau 63). Fundir não economiza nível. Tabela por refresh:
  `scripts/em9_refresh_ctx.txt`.
- **Busca nas aproximações** (`scripts/search.py`, `kl_eval.py`, `sm_split.py`):
  - cortar o grau da GELU explode o KL: as camadas 2 e 11 têm xmax 82/70 por outliers, e grau 127 → KL 2e-2;
  - a LayerNorm no grau 15 não tira bootstrap;
  - a alavanca era a softmax (40 de 101 refreshes): divisão l1/l2 por camada mais a faixa da rodada 0 calibrada;
  - subagente (`subagent/ganhos-rapidos.md`): exp com minimax **relativo** (smrel), graus da GELU 1 nível mais rasos
    (gelubs) e normalização adiada da última rodada da softmax (branch `defer-norm`, `SoftmaxCfg.defer_norm` por
    camada). Normalização adiada na camada 11 deixa o CutMax inviável ("sine bent past its first zero"): adiar só nas
    camadas 1 e 3.
- **Auditoria** (`subagent/audit/auditoria.md`): NO-GO para o plano de 98.
  - Primeiro bloqueador: os pousos do recorde estavam fixos no compilador e no gêmeo de CPU; o plano não compilava.
  - Segundo bloqueador: as faixas ×4/×6 divergiam na cauda (KL = inf em 5 janelas de estresse).
  - Outros achados: o `headroom_split.py` escolhia faixas olhando o dev (proibido; removido) e o Prefill quebrava sem um
    switch por plano.
- **Máquina:** 15 GB de RAM. Uma ILP por vez, jobs pesados sob `systemd-run --user --scope -p MemoryMax=…`. Rodar 4–5
  ILPs mais uma captura em paralelo estourou a memória (OOM) em 2026-10-27.

## 2026-10-27/28 — correções da auditoria (tarefas 1-3)

### Tarefa 1: EvalMod atrás de um switch, plano compila — FEITA

Commit `d8a7643` em `work/perseus` (branch `evalmod-d27`, sobre `d270f7a`). Campo tipado `bts.evalmod`
(`FIDESLIB_BTS_EVALMOD`, linha `runtime=` do plano): sem valor/`shipped` = tabelas do recorde (grau 14 / grau 20,
5 ângulos duplos, 10 níveis) — de novo o padrão; `d27` = grau 27 + 4 DA (9 níveis). `pc_core::bts_setup::EvalMod`
dá tabela, DA e níveis; os pousos (compilador `runtime_shadow`, Shadow runtime/fakecap do pc-py, `fakecap.py` via
`_pc.fakecap_landings`, `high_shapes` do CCMM) saem de `EvalMod::levels` (`BtsConfig::shift_evalmod`,
`multi_stage_landing(.., em)`): toda rota menos a SPRU (s = 1) pousa 1 nível (2 primos) mais raso. As configs da
GPU (dense, StC-first, SparseB, clássica) pegam a tabela do Config; `pc-bench`/`pc-check` leem o env. O campo é
load-bearing (amarra o plano). `em_depth` sem argumentos lista as duas séries com os DA certos.

Critério (compile_check com os 14 estágios encadeados como no `prepare()`: grafo dos blocos 0-12 + bloco 13 do
`kc16_cm`, em `work/graphs_combo/`):

| plano | switch | erros |
|---|---|---|
| recorde `gpt2_decode_python_n32_kc16_cm` | desligado | **0** |
| `fin_dfr13_m_cm_d27` (= `fin_dfr13_m_cm` + `FIDESLIB_BTS_EVALMOD=d27`) | d27 | **0** |
| recorde | d27 | 34.085 (prova que o compilador vê os pousos novos) |
| `fin_dfr13_m_cm` (sem switch) | desligado | 32.557 |
| Prefill-128 recorde `gpt2_prefill128_pfbhrd_cm` | desligado | **0** |

Compilar o bloco 13 sozinho dá 1.006 erros (lê o nível de entrada da captura do recorde); encadeado, 0.
`cargo test` (pc-core, pc-config, pc-shadow, pc-program, pc-client): tudo passa menos os `oracle_*` (dump privado).
`route::tests::multi_stage_landings` passa e cobre d27; teste novo `evalmod_series_levels` confere com o
`build_evalmod` real que cada série gasta `levels()` níveis. `shipped_stamps_round_trip` só falha no checkout de
trabalho por causa de um plano experimental não versionado (`fin_dfr_m3_cm`, knob `PLAN_B12_ENV_CAP`); num
worktree limpo (`work/perseus-t1chk`) passa. pytest CPU: 414 passam; as 4 falhas de `test_gpt2_primitives_cpu.py`
(GELU THOR do `gpt2_base_n32`) falham igual no worktree do recorde.

Consequência para a receita d27: com o fakecap corrigido o push K/V passa a ser capturado em 24 (antes 26), então o
remap `26:24` sai e o `24:14` herdado do recorde (no-op no recorde: nada pousa em 24) PRECISA sair também, senão
manda o push K/V para 14.

### Tarefa 2: faixas da rodada 0 da softmax recalibradas — FEITA

`scripts/sm_split.py` reescrito (o v1 foi removido em 2026-11-01; está no histórico do git). Agora usa a regra do recorde
(`scripts/utils/fit_gpt2_decode16.py::fit_softmax`), que a docstring antiga prometia e a linha 73 não aplicava:
faixa = divisores dos dados ×3 nos dois lados (cabeças com padding incluídas), com topo ≥ 1,05·exp((clip_hi − mid)/2^l2),
isto é, toda query dentro da faixa de scores converge, independente dos dados. Esse topo estrutural é o que deixa
`l2 = 0` caro onde a faixa é larga. A contagem de níveis já desconta a quadratura que o `smrel=0` tira depois.
Janelas (`scripts/owt_windows.py`), metades disjuntas do pool OWT:
- calibração: 865 inícios de documento + offsets aleatórios em [0, 1M), 16.384 janelas, sem as 5 de estresse;
- held-out: 898 inícios + aleatórios em [1M, 2M), 17.282 janelas.

**Nenhuma escolha olhou o dev.** `headroom_split.py` (que usava o dev) foi substituído por `scripts/band_check.py`,
que lê a faixa da própria config (`band_from_seed`). A config sai de `scripts/build_cfg.py NOME SPLIT`, o mesmo
pipeline do `cand_dfr13_cm`: split → smrel=0 → gelubs → defer 1,3 → cutmax do kc16_cm. Com o split antigo ele
reproduz `cand_dfr13_cm` bit a bit.

Resultado `cand_rc3` (`work/sm_split_rc3_t0.01.json`, alvo final 1e-2): **nenhuma camada escolhe l2 = 0**.
Splits l1/l2 por camada: 1/3, 1/2, 2/1, 1/2, 1/4, 2/2, 2/2, 2/2, 2/1, 2/1, 1/2, 2/3. Softmax 204 níveis/token,
contra 210 do recorde com o mesmo desconto do smrel. O ganho da softmax encolhe de −21 para −6, como a auditoria
previa.

| held-out (17.282 janelas × 16 posições) | janelas fora da faixa | folga mín. embaixo | folga mín. em cima |
|---|---|---|---|
| recorde `kc16_cm` | 0 | 2,57× | 5,79× |
| antigo `cand_dfr13_cm` (×4/×6) | **17** (incl. 1094059, 1149233, 1678699) | 1,20× | **0,08×** (L10) |
| **novo `cand_rc3`** | **0** | **2,79×** | **5,31×** |

KL float64 (`subagent/audit/kl_windows.py`, agora com `KLW_CFG`) nas 5 janelas de estresse:

| janela | `cand_rc3` | recorde | antigo |
|---|---|---|---|
| 1094059 | 5,8e-6 | 2,9e-6 | inf |
| 1149233 | 4,1e-7 | 2,4e-7 | inf |
| 1678699 | 4,3e-6 | 2,7e-6 | inf |
| 257792 | 2,9e-6 | 1,3e-6 | inf |
| 853437 | 3,1e-6 | 1,0e-6 | inf |

Todas finitas, ≈2× o recorde: a mesma razão de antes fora das janelas patológicas (exp relativa, GELU rasa, alvo
1e-2). KL nas 8 janelas OWT de sempre: 4,2e-6 (máx 1,0e-4). O `sa_s6t2r0gb` dava 4,1e-6.
Custo: cada passada de 16k janelas em float64 leva cerca de 55 min nesta máquina.

### Tarefa 3: plano refeito — FEITA: `plans/fin_rc3b_m_cm` = 98 + 12 = **110/token** (recorde 120, −8,3 %)

`work/final_plan.sh rc3b configs/model/approximation/cand_rc3/configs.json` (`JOBS=1 MERGE_MEM=5G`). Com
`JOBS=2` a captura fundida morreu duas vezes pelo OOM do sistema: os apps do usuário ocupam ~9 GB. Mudanças no
script (o original está em `final_plan_v1.sh`):
- `FIDESLIB_BTS_EVALMOD=d27` só na linha `runtime=`. As capturas ficam sem switch, porque a captura do CutMax
  (estágio 13) é a do recorde e o carimbo dela tem de ser igual ao do estágio 12 (`make_cutmax_plan.sh`).
- `PLAN_DELIB_LANDING_REMAP=26:24` no lugar de `24:14,26:24` (ver o bug abaixo).

**Bug novo, que também invalida o plano de 98 da sessão anterior.** O remap dos pousos deliberados é aplicado DUAS
vezes na carga do grafo, de propósito, no planner (`ir.py`) e no compilador (`pc-shadow/src/graph.rs:407`, "the raw
round trip ... from_nodes remaps it again"). Com `24:14,26:24` o push K/V capturado em 26 vira 24 e depois 14, o
pouso da SPRU. `route_for_landing` então põe os 12 pushes K/V na rota SPRU (s = 1), que reduz o vetor a um slot: na
GPU, lixo. O `compile_check` dá 0 erros mesmo assim, porque planner e compilador leem o mesmo pouso remapeado. Novo
`scripts/route_mix.py` compila os 14 estágios e compara a rota de cada refresh deliberado com a do recorde:
`fin_dfr13_m_cm` (o "98", removido em 2026-11-01) e `fin_rc3_m_cm` (primeira tentativa, 105) têm 12 fora; `fin_rc3b_m_cm` tem 0. Toda
contagem com a receita em9 (em9 sozinho 101/113, a tabela de ILPs de 2026-10-27, 86/98) saiu com esse bug:
otimista (a SPRU pousa mais raso) e com 12 dense trocados por SPRU.

| checagem do `fin_rc3b_m_cm` | resultado |
|---|---|
| compile_check, 14 estágios encadeados (`work/graphs_combo/rc3b`) | **0 erros** (sem o switch: 33.320) |
| deliberados fora da rota do recorde (`route_mix.py`) | **0** |
| mix blocos 0-12, dense / SPRU / SparseB-512 | 36 / 25 / 37 (recorde 36 / 25 / 47) |
| CutMax (estágio 13) | 12, saída 48:2 (igual ao recorde) |

Todo o ganho são 10 refreshes SparseB-512. Pela tabela `bts_ms` da auditoria (512 = 17,39 ms) isso dá ≈ −174 ms
dos bootstraps. Contra o ganho: EvalMod dense ~20 % mais caro (48 dense/token) e pousos mais ricos. O ganho real
fica em poucos %, e só a GPU decide se passa a margem de 1 %.

R6 da auditoria revisto: o ×1004 / ×168 da captura fundida é `v_28` de `blk5.ln_1` / `blk10.ln_1` (LayerNorm). O
máximo fundido dele é o mesmo da captura do recorde (`kc16m`: 1,51), ou seja, crescimento posição 0 → 16 posições
que o recorde também tem. Não vem de l2 = 0.

Código: `perseus-evalmod-switch.bundle` (a3dff85..evalmod-d27 = 67717aa, d270f7a, d8a7643) e
`0001-EvalMod-degree-27-tables-behind-FIDESLIB_BTS_EVALMOD.patch`. O bundle antigo ficou como estava.

### 2026-10-28 — teste de sanidade (subagente cético): as 6 alegações CONFIRMADAS
Relatório: `subagent/sanity/relatorio.md`.
- Sem o switch, o código é igual ao do recorde.
- Com `d27`, a série consome sempre 2 primos a menos em todo topo, nl e K. Varredura com o `build_evalmod` da GPU:
  0 divergências.
- Nenhum ponto de runtime ficou de fora.
- Plano: 0 erros, 98 + 12, deliberados na rota do recorde. O push K/V é dense StC-first, pousa em 24.
- Remap duplo e "SPRU no K/V = lixo": confirmados, e nenhuma checagem recusa esse caso.
- Fórmula do divisor do `band_check` = a do caminho cifrado. O gêmeo difere menos de 0,5 %.
- Configs só diferem onde devem.

### 2026-10-28 — NOVO RECORDE no upstream (`origin/speedrun` = d97a6f7)
- **Decode-8 recorde 4, LeanChainDecode: 1,798 s/token** (recorde 3: 2,325), PR #14 do erodola, 2026-10-27.
  - Cadeia `n32a4`: 4 dígitos de key switching em vez de 6; as entradas do StC-first em primos simples.
  - Softmax centrada numa chave de referência pública por cabeça (blocos 0-7, 9, 10): 28 rodadas de normalização em
    vez de 36, e 8 SparseB a menos. Plano 114 refreshes/token.
  - Config `gpt2_base_n32_kc16r_cm`.
- **Prefill-128 recorde 4, LeanChainPrefill: 10,12 s.**
- Para a nossa entrada: rebase em d97a6f7. O switch do EvalMod precisa ser portado para a cadeia nova (route.rs,
  compile/mod.rs e shadow.rs mudaram). O ganho das faixas da softmax provavelmente se sobrepõe ao da softmax
  centrada (os dois cortam SparseB da softmax). Recalibrar sobre o `kc16r` e refazer o plano antes de comparar.

### 2026-10-28 — vale a pena sobre o recorde 4? NO-GO para o em9 (subagente, `subagent/r4/relatorio.md`)
- O em9 ainda poupa 1 nível por bootstrap na `n32a4` (medido com o `build_evalmod`). Porte em `work/perseus-r4`,
  branch `r4-em9`, commit fc19e09: 4 conflitos triviais.
- **ILP com a receita em9 no recorde 4: 114 → 113/token** (só 1 SparseB-512 a menos). 0 erros, deliberados na rota
  do recorde. Tempo estimado entre −0,1 % e +1,2 % (EvalMod dense +20 % mais caro). Não passa a regra 5 (≥ 1 %).
- O min-cut não serve para diferenças: deu −4/−9 onde a ILP deu −1.
- Na softmax centrada do recorde 4, split/faixas valem 0 (a faixa da rodada 0 já é exata). Sobra o smrel (exp
  relativa, com a série valendo 1 no topo): −12 níveis/token, KL neutro, ~−2 refreshes no proxy. Sem ILP ainda.
  gelubs não compensa. defer_norm não testado. smrel e defer não dependem do em9.
- Bug de planner com a receita em9: no estágio 12 o `PLAN_ILP_EXIT_CAP` não segura a saída (46/d2).
- Caminho barato, se quiser continuar: 2 ILPs na posição 0 (recorde 4 contra recorde 4 + smrel). Captura fundida só
  se der ≥ 2 refreshes, e ela não cabe nesta máquina (servidor).

### 2026-10-28 — servidor (`sandbox-keml`): setup interrompido
- Uso autorizado pelo usuário, com limites: até 12 núcleos, RAM bem abaixo do total (watchdog com teto de 32 GB),
  sempre em tmux.
- O setup foi parado a pedido, no meio do build do `_pc`: outra pessoa começou a usar a máquina.
- Ficaram no servidor, intactos: `~/pesquisa/perseus-speedrun/` (clone `work/perseus-r4` no branch r4-em9, venv, build
  parcial, scripts) e `~/.cargo` / `~/.rustup`.
- Para retomar: `tmux new -s perseus`, rodar `bash watchdog.sh` numa janela e `bash setup.sh` noutra (o setup pula o
  clone e o venv já feitos). Depois `bash subagent/r4/ilps.sh`, com as duas ILPs da posição 0 em paralelo.
- Cópia local do pacote: `subagent/r4/` (plan_count_r4.sh). O resto foi gerado no scratchpad da sessão e está no
  servidor.

### 2026-10-29 — VRAM necessária (medida na GTX 1650 da zocal, `work/sm75/`: `pc-bench` para sm_75)
- `pc-bench bts-prec --route dense`: o MemPlan pede **31,4 GB** com `--k 1` ou `--k 8` (chaves 14,7 GB = 75 rotações,
  `bts.work` 8,1 GB, `bts.dense` 4,9 GB, `ks_work` 1,9 GB). Uma placa de 32 GB fica no limite ⇒ na prática ≥ 40 GB.
- Rotas 1 / 256 / 512 do `pc-bench` (`crates/pc-tools/src/bts_sparse.rs`) leem segredo e chaves do dump privado dos
  autores (`G0_DEFAULT` = `/data02/users/mithrillo/.../dumps/g0`) ⇒ não rodam sem ele, em GPU nenhuma, a menos que se
  escreva um gerador de dump falso.
- Plano inteiro: ≥ 48 GB (README do upstream). As 3090 do hsr (24 GB) não rodam nem o `bts-prec` dense.
- Kernels isolados na 1650 (zocal `~/perseus-sm75/`): `pc-kernels-selftest` = 0 divergências (139 checagens ×
  u32×Barrett64/Barrett32, sm_75). `pc-bench modmul --k 1,2` (build com `--features pc-kernels/mont`; precisou tirar o
  piso K ≥ 8 do modo bench em `crates/pc-tools/src/word_seam.rs:97`, mudança local SEM commit no `ln-defer`): as 3
  reduções modulares empatam (≤ 1 %), saídas idênticas; tudo DRAM-bound (add K=1 no topo 53 ≈ 172 GB/s, NTT ≈ 145 GB/s
  = pico da GDDR6 da 1650); rotação/KS ≈ 7,3 ms ≈ 15× um add. Tempo da 1650 não transfere p/ Blackwell (lá o gargalo
  é L2, Theodosian). Serve como bancada de CORREÇÃO de kernel, não de desempenho. Sem `ncu` na zocal.

### 2026-10-29 — d27 só na SparseB (`bts.evalmod = d27sb`): NO-GO, o em9 em todas as rotas vence
- Switch novo `EvalMod::D27Sparse` ("d27sb", working tree do `ln-defer`, SEM commit): SparseB / sparse clássico com a
  tabela d27, dense / StC-first com a do recorde (`dense_route()` / `sparse_route()`; pousos por família em
  `BtsConfig::shift_evalmod`, fakecap, compile). Teste `route::tests::multi_stage_landings` estendido: passa.
  Receita: só `PLAN_SPARSE_BTS_OUT=1:16,512:24` (`MODE=d27sb` em `subagent/r4/plan_n32a4.sh`).
- Rodado na zocal (CPU, `work/lndefer/run_d27sb.sh`, ILP, 24 min, MemoryMax 3,5 G): `lnd_fin_e3_d27sb_cm` =
  **99 + 13 = 112/token** (em9 em todas as rotas: 95 + 13 = 108; recorde 4: 114). compile_check 0 erros (14 estágios),
  route_mix 0 fora da rota; estágio 12 saiu em 44/d2 pelo ILP (sem o bug do em9).
- time_model (bootstraps): recorde 1429,9 ms; em9 1344,3 ms + custo do EvalMod dense d27 (+12-36 ms) ⇒ −2,8 a −4,1 %
  do token; **d27sb 1401,4 ms ⇒ −1,6 %**. Mix SparseB-512: recorde 40, em9 34, d27sb 38 ⇒ 4 dos 6 SparseB cortados vêm
  dos pousos MAIS RASOS da rota dense (folga a jusante), não da tabela da SparseB. Manter o em9 em todas as rotas.

### 2026-10-29 — smrel por cima da LN adiada (e3) + em9: 0 bootstraps ⇒ NO-GO
- Capturas AQUI: posição 0 `graphs/lnd_smrel_e3` (`LN_DEFER=1 CFG=.../r4_smrel/configs.json bash work/lndefer/cap.sh`,
  KL = o de sempre) e fundida `graphs/lnd_smrel_e3m` (`BASE=… PLAN=bootstrap_placements/capplan_r4_smrel JOBS=1`,
  MemoryMax 7 G, 21 min). `capplan_r4_smrel` = só as linhas runtime= do recorde 4 e configs= do smrel.
- Plano na zocal (`work/lndefer/run_lnd_plan.sh`, MODE=em9): ILP do estágio 12 saiu em 46/d2 (bug conhecido do em9)
  ⇒ `RESUME=1 PLACER=mincut B12SUF=mc`. `lnd_fin_e3_smrel_em9_cm` = **95 + 13 = 108/token, contagem por bloco
  IDÊNTICA à do `lnd_fin_e3_em9_cm`** (placements diferentes em 12/13 blocos), mesmo mix e mesmo time_model (1344,3 ms).
  compile_check 0, route_mix 0. Os −12 níveis da softmax não viram bootstrap (quantização, como a LN adiada sozinha).
- `subagent/r4/plan_n32a4.sh`: agora grava `configs=$CONFIGS` no PLAN_CMD (o CutMax recusava) e é seguro para
  RESUME repetido (antes o 2º resume sobrescreveu `lnd_fin_e3_smrel_em9_b12mc/block_12_placement.blocks_pass.json`
  com a cópia do min-cut: o bloco 12 da passada ILP dos blocos se perdeu; não é usado pelo plano).
- PENDENTE p/ GPU (vale para TODOS os planos lnd_*): o PLAN_CMD não grava `LN_DEFER=1` no runtime= nem o grafo
  `lnd_*` na linha graph= (copia os do recorde) ⇒ servir pelo PLAN_CMD rodaria SEM a LN adiada. Resolver antes da GPU
  (e ver se o parser de config aceita a chave).
- Melhor plano continua `lnd_fin_e3_em9_cm` (108/token, −2,8 a −4,1 % estimado).

### 2026-10-29 — preparação da GPU: plano servível + script único
- `LN_DEFER` / `LN_DEFER_EXIT` viraram campos do pc-config (`model.ln_defer`, `model.ln_defer_exit`): antes a linha
  `runtime=` com `LN_DEFER=1` era RECUSADA ("no field for env name"), e sem ela o servidor regrava o grafo
  (`PlannedRun.prepare` grava o fluxo do modelo e compila contra o plano) com a LayerNorm do recorde.
  `work/lndefer/check_runtime.py PLAN`: antes `LN_DEFER=None`, depois `model.ln_defer=True` + `LN_DEFER='1'` no env.
  PLAN_CMD de `lnd_fin_e3_em9{,_cm}` corrigido (runtime + `graph=graphs/lnd_on_e3m`); compile_check 0, route_mix 0.
  `plan_n32a4.sh` ganhou `RT_ADD=` (ex.: `RT_ADD=LN_DEFER=1`) e grava `graph=$GR`.
- `run_decode --fake --plan` está QUEBRADO no upstream (`CapCore` sem `parse_bootstrap_plan_file`) ⇒ não há gêmeo de CPU
  do plano inteiro; o teste de ponta a ponta só na GPU.
- Commits locais no `ln-defer` (sobre r4-em9 fc19e09): 7e497ca (d27sb + campos LN_DEFER; teste `field_table`
  atualizado 260→262 / 352→354), 98e0769 (pc-bench modmul), 44858c8 (plano `lnd_fin_e3_em9_cm` + grafos
  `lnd_on_e3{,m}` + `speedrun/server.py` DECODE_PLAN). `check_pr.py --base origin/speedrun`: passa.
  Bundle: `perseus-lndefer-entry.bundle` (origin/speedrun d97a6f7..ln-defer 44858c8, 2 MB).
- `gpu/run_gpu.sh` (NÃO testado, sem GPU aqui): setup do zero (clone + bundle + worktrees cliente 89bc830 / lean
  24d4b3f / recorde d97a6f7, venvs, export dos pesos), `bts` (pc-bench dense shipped × d27: precisão e ms), `decode`
  (6 pares recorde/entrada intercalados pelo harness + score.py + pico de memória), `prefill` (1 par: tem de rodar).
  Fica de fora: KL nas 5 janelas de estresse e held-out C4 (precisam de oráculo + run planejado fora do harness).

### 2026-10-29 — teste de sanidade (subagente Opus xhigh): GO condicionado; script da GPU corrigido
- Relatório: `subagent/sanity2/relatorio.md`. Reproduziu na CPU o `prepare()` do servidor (modo de pouca memória, com o
  recorde como controle): compile 0 erros, 108, mesmas 124 chaves, 16 posições ok no `precompute`. Riscos que ficam:
  a série d27 nunca rodou em GPU (R1); o servidor grava com os pousos d27 e a captura foi feita com os do recorde ⇒
  entrada do bloco 0 cai de 26 p/ 24 (C1, compila, custo pequeno); ganho só de modelo (R2). Máquina: VM/bare metal
  (bwrap + sessão systemd de usuário + CUDA 13 sob /usr), não container.
- `gpu/run_gpu.sh` reescrito: etapas preflight | setup | bts | smoke | decode | prefill | all; falha de run detectada
  (antes o errexit morria dentro de `$(...)`), score.py nunca aborta o script (veredito no log), prefill com 2 pares,
  setup re-executável e conferindo HEAD = `ENTRY_SHA`, preflight real (≥ 48 GB, nvcc 13.x sob /usr, bwrap, systemd-run),
  run isolado da entrada (`smoke`) antes dos pares, 4 prompts dev por run com offset rotativo (24 prompts em 6 pares),
  trap do nvidia-smi. Testado com mocks (sucesso, falha no 3º run, score "não é record") e a parte git num clone.
- Commit 83b8d55 (teste Python: 262 campos); bundle regenerado (origin/speedrun..83b8d55).
- Incidente: um `cd` falho fez um `git reset --hard d97a6f7` rodar no worktree `work/perseus-r4`; o mesmo comando
  restaurou o `ln-defer` pelo bundle (reflog 18:36:26); nada perdido (tudo commitado). Só os mtimes mudaram.

### 2026-10-29/30 — Runpod NÃO roda o harness oficial (testado, ~US$ 0,13)
- Pod `2uitbzbe5q58e6` (RTX PRO 6000 WK, Community, US$ 1,69/h, imagem `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`),
  terminado após ~5 min. GPU/driver/nvcc OK (97887 MiB, driver 580.65.06, nvcc 13.0 em /usr/local/cuda-13.0), 37 vCPU / 167 GB.
- PID 1 = `docker-init` (container, sem systemd), seccomp do Docker ativo, sem CAP_SYS_ADMIN:
  `bwrap --unshare-user ...` ⇒ "No permissions to create new namespace"; `unshare -U` ⇒ "Operation not permitted";
  `systemd-run --user ...` ⇒ "Failed to connect to bus". Os dois mecanismos do `speedrun/runner/sandbox.sh` falham.
- ⇒ para os `run.json` do harness: provedor com VM/bare metal. Runpod serve só p/ A/B fora do harness (não reportável).

### 2026-10-30 — A/B fora do harness, pronto para a Runpod (`NOSANDBOX=1`)
- `gpu/ab/sandbox.sh`: stand-in do `speedrun/runner/sandbox.sh` (mesmos argumentos; ambiente vazio, 1 GPU visível via
  CUDA_VISIBLE_DEVICES, cwd/HOME próprios, `/out` remapeado, timeout, sessão própria ⇒ stop mata a árvore inteira).
  SEM isolamento/limites de memória/seccomp. `run_gpu.sh` com NOSANDBOX=1: preflight pula bwrap/systemd; setup monta
  `$ROOT/ab/speedrun/{runner,harness,prompts}` (cópia do recorde 4, fora dos checkouts) com o stand-in e o `stop()` do
  run.py terminando o processo; runs vão para `$ROOT/out-ab`. Mesmo relógio, protocolo, KL e score.py do harness.
- Os run.json saem com `harness: null` ⇒ `score.py --official` RECUSA (testado). NÃO são números do harness: não vão p/ PR.
- Testado aqui com o dummy_server: sessão completa ok (mesmas chaves de run.json que o harness original com bwrap),
  servidor travado ⇒ void em 3 s + grupo de processos encerrado (sem órfãos).
- Uso no Pod: `NOSANDBOX=1 BUNDLE=… bash run_gpu.sh all` (preflight → setup → bts → smoke → decode 6 pares → prefill 2).

### 2026-10-30 — A/B na GPU (Runpod, RTX PRO 6000 WK, fora do harness): NENHUMA variante é recorde
Resultados em `gpu/results-2026-10-30-runpod/` (23 run.json, score_*.txt, logs de bts/build/servidor). `NOSANDBOX=1`.
| comparação (mesma placa, alternada) | entrada | recorde 4 | razão | KL médio |
|---|---|---|---|---|
| Decode, LN(e3)+em9 `lnd_fin_e3_em9_cm` (108 refreshes), 6 pares/24 prompts | 1,768 s | 1,690 s | **×0,956 (4,4 % mais lenta)** | 2,5e-6, argmax 100 % |
| Decode, LN(e3)+d27sb `lnd_fin_e3_d27sb_cm` (112), 3 pares/12 prompts | 1,694 s | 1,688 s | ×0,997 (empate) | 6,2e-6, argmax 100 % |
| Prefill-128 (switches desligados), 2 pares | 9,33 s | 9,328 s | ×1,000 | 2,7e-5 |
- Causa (pc-bench na mesma GPU, n32): EvalMod d27 é ~27 % mais caro na etapa EvalMod ⇒ +3,0 ms/refresh na rota dense
  (21,94→24,93 ms K=1) e +2,2 ms na StC-first (18,40→20,61 ms K=1; +2,6 ms/ct K=8). 42 refreshes dense/token ⇒ ~+100 ms,
  maior que a economia dos 6 SparseB. O time_model (+20 % só no EvalMod, 12-36 ms) SUBESTIMOU o custo. Precisão d27 na
  dense: igual na parte real, +3-4 bits na complexa; pousa 2 primos mais raso (confirmado).
- LN adiada funciona na GPU (KL 10× MENOR que o recorde), mas sozinha + 2 SparseB a menos (d27sb) não muda o tempo.
- Lição: contar bootstraps não basta; o custo por bootstrap da série nova manda. A linha "EvalMod d27" está FECHADA
  como fonte de ganho de tempo; qualquer próxima ideia de níveis precisa medir o custo por refresh na GPU antes.
- Pods: 2 testes de sandbox + 1 A/B interrompido + 1 A/B completo, todos terminados. Custo total ≈ US$ 2,8.

### 2026-10-30 (tarde) — modelo de tempo recalibrado e A/B na GPU: `r4_ln3_sbd27d1_cm` ×1,019 (fora do harness)
Relatório completo: `tmodel/RELATORIO-2026-10-30.md`. Resumo:
- Modelo calibrado nas etapas do `pc-bench` (erro ±6–15 % por etapa). Ele explica os dois A/B da manhã com o mesmo
  desvio de ~+9,5 ms, que depois se mostrou ser custo da LN adiada.
- Onde vão os 1690 ms do recorde 4 (WK): refreshes ~65 %, ops ~26 % (key switch ~290 ms; LayerNorm 100, softmax 74,
  GELU 42, CutMax 40), resíduo ~9 %. Um nível a mais ou a menos em todas as ops vale ±47 ms.
- A/B na RTX PRO 6000 Max-Q (3 pares alternados, `NOSANDBOX=1`, `harness: null`):
  - `r4_ln3_sbd27d1_cm` (opções C + A): 2,324 s contra 2,369 s, **×1,019**, p 0,0003, KL 4,5e-6, score.py "NEW RECORD";
  - `sbd27d1_cm` (opção A): ×1,007.
  - Em proporção, o modelo acertou os dois (−1,7 % e −0,55 % previstos).

### 2026-10-30 (noite) — ideia do Trefethen (ATAP): EvalMod PAR em u' = u − 1/4 (só modelo, sem GPU)
- O alvo pré-ângulo-duplo `s0 cos(2π(u−1/4)/2^r)` é PAR em u' = u − 1/4 e a união de intervalos {I+δ−1/4} é quase
  simétrica (|u'| ≤ K−3/4). Série par: `p = Σ c_k T_k(z)`, z = 2((u−1/4)/K)² − 1 (AddConst + 1 quadrado + ×2 − 1),
  grau dz = metade ⇒ mesma profundidade (1 + D(dz) + r) com menos Mult/Rescale/WSum. Os fits d27 já tinham o grau 27 = 0.
- `scripts/em_even.py`, `scripts/em_even_scan.py` (bits), `crates/pc-shadow/examples/em_hist.rs` (no `work/perseus-r4`, não
  commitado: histograma de ops + soma de limbs; 3º argumento K liga o pré-passo par). Fits em `fits/even_*.json`.

| K=16, 9 níveis (18 limbs) | bits | Mult | Rescale | Σlimbs mult / resc / wsum |
|---|---|---|---|---|
| recorde d14+5DA (10 níveis) | 27,4 | 10 | 18 | 428 / 700 / 522 |
| d27+4DA (em9, medido +27 % no EvalMod) | 43,5 | 12 | 23 | 514 / 902 / 916 |
| **par z11+4DA** | **34,1–35,3** | 9 | 15 | **400 / 582 / 410** |
| par z13+4DA | 44,4 | 11 | 23 | 472 / 896 / 304 |

⇒ par z11 tem EvalMod MENOR que o recorde (previsão −10…−17 % só no EvalMod) e precisão maior. Ele deve ser usado com
+1 raise drop, para pousar como o recorde 4; com os planos em9 o pouso sobe e as ops encarecem ~+52 ms (`ATAP-ganhos.md`). SparseB K=24: par z13+4DA 9 níveis = 30–33 bits (d27 geral: 28,2; recorde 35,2). Falta: o pré-passo
em `build_evalmod` + variante `EvalMod`, KL e `pc-bench bts` na GPU (o time_model errou no d27: medir antes).


### 2026-11-01 — revisão das ideias do Trefethen (ATAP) e limpeza dos `.md`
Detalhes em `ATAP-ganhos.md`. Nada rodou em GPU.
- **GELU par:** o BSGS atual já pula os termos ímpares (`IMPL_DEAD_OPS`). O ganho real é 210 → 180 produtos
  ct×ct/token (~−4 ms), não −170 produtos. A contagem é do novo `scripts/gelu_bsgs_count.py`, que reimplementa o BSGS
  do upstream sem importar o Perseus. `IDEIA-gelu-par.md` foi reescrita.
- **EvalMod par:** usar o nível livre num raise drop, para manter os pousos e o plano do recorde 4. Estimativa: −25 a
  −45 ms/token. A varredura de séries pares de grau baixo com 5–6 DA não achou nenhuma que bata o recorde em 9 níveis.
- **Softmax por bandas:** com erro relativo, as bandas viram o minimax relativo da semente, ou seja, o `smrel`, já
  medido. O (d=3, k=3) da proposta tem erro relativo 8,9e-2 (o recorde tem 1,1e-4). Linha fechada.
  `IDEIA-softmax-ponderada.md` foi reescrita.
- Removidos por obsoletos:
  - `HANDOFF.md` (recorde 3, tarefas feitas);
  - `plans/fin_dfr13_m_cm` (o plano de 98, inválido: K/V na SPRU);
  - `plans/gpt2_decode_em8` (receita com o bug do remap, nunca completada);
  - `scripts/headroom_split.py` (usava o dev, substituído por `band_check.py`);
  - `scripts/sm_split_v1.py` (não aplicava o topo do recorde).

  As seções do recorde 3 com números invalidados viraram o resumo acima. Tudo isso continua no histórico do git.

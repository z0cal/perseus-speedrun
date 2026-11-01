# Teste de sanidade 2: entrada Decode-8 `ln-defer` (44858c8, plano `lnd_fin_e3_em9_cm`), antes de alugar a GPU

Data: 2026-10-29. Máquina: zocal (CPU, sem GPU NVIDIA). Cada job pesado rodou sob
`systemd-run --user --scope -p MemoryMax=6G nice -n 10`, um por vez. Não houve rede, servidor, commit nem `rm`.
Nenhum arquivo rastreado de `work/perseus-r4` foi alterado (`git status` limpo, fora os untracked que já existiam).

## Veredito: GO condicionado

O código e o plano passam em todas as checagens que dá para fazer na CPU. Isso inclui o risco principal, que
nunca tinha sido testado: o `prepare()` completo do servidor (gravação do gêmeo de CPU, compile contra o plano,
pedido de chaves e amarração das 16 posições). Ele foi reproduzido agora, com o recorde como controle.

Antes de pagar, faça duas coisas (minutos de trabalho):

1. Corrigir os bugs B1-B3 do `gpu/run_gpu.sh`. Eles não corrompem resultados, mas queimam horas de GPU e pulam o
   prefill.
2. Alugar uma **VM ou bare metal**, não um container. Precisa ter sessão systemd de usuário, `bwrap` funcionando
   (user namespaces sem privilégio), CUDA 13 sob `/usr` (o sandbox só monta `/usr`) e ≥ 48 GB de VRAM.

Riscos que só a GPU resolve: a série d27 nunca rodou em GPU nenhuma (R1), e o ganho é de modelo, não medido (R2).

## As 8 alegações

| # | Alegação | Status | Evidência |
|---|---|---|---|
| 1 | Switches desligados = recorde 4 | **CONFIRMADA** | Diff de 27 arquivos revisado linha a linha. `EvalMod::Shipped` devolve exatamente as tabelas/ângulos de antes (`EVALMOD_CHEB`, `SPARSE_B_CHEB_K24`, 5 = `EVALMOD_DA_ITS` = `SPARSE_B_DA_ITS`; `ps_k24()` = `from_coeffs(K24)`). Deslocamento 0. `fakecap_landings` default = `parse_landings` antigo (`cfg_check.log`). **Empírico:** o gêmeo do servidor no HEAD, com o plano do recorde, grava fluxos canonicamente IDÊNTICOS ao grafo do recorde `gpt2_decode_python_n32_kc16rm` nos 14 blocos (0 nós diferentes, args incluídos). Compila em 101 + 13 = 114 com 0 erros e pede as mesmas 124 chaves (`prep_record.log`). Única mudança default: o pc-bench modmul não força mais K ≥ 8 no modo só-bench (ferramenta, intencional). |
| 2 | `D27Sparse`: nenhum `panic!` alcançável; pousos por família coerentes | **CONFIRMADA** | Todos os chamadores de `dense()/sparse_k24()/da_its()/levels()` passam por `dense_route()/sparse_route()` ou usam só Shipped/D27 (`em_depth`, teste `evalmod_series_levels`). Empírico com `d27sb`: `Session.key_manifest` (StC-first, dense, SparseB, clássica, high shapes) OK, com as mesmas 124 chaves de None/d27; o valor inválido "bogus" dá `ConfigError` (`keyneeds_modes.log`). O plano `lnd_fin_e3_d27sb_cm` compila com 0 erros, 99 + 13 (`d27sb_compile.log`). `em_depth` roda. Famílias coerentes: s=0, `bootstrap_level`, `stc_first_fallback` e `dense_shipped` vão por `dense_route`; s=1 (SPRU) fica intocado; os demais vão por `sparse_route`. Isso vale em `shift_evalmod`, `fakecap_landings_of`, `runtime_shadow` (desloca os defaults ANTES das sobrescritas da receita, sem deslocar duas vezes), `BtsSetup/ClassicCfg::set_evalmod`, `SparseBCfg.evalmod`, `high_shapes`. Testes `multi_stage_landings` e `evalmod_series_levels` passam no clone. |
| 3 | `model.ln_defer/ln_defer_exit`; `LN_DEFER` exportado a tempo | **CONFIRMADA (com ressalva)** | `pcfg.plan_runtime` aceita o PLAN_CMD (`model.ln_defer=True`, `bts.evalmod='d27'`). A Config da sessão difere da do recorde só nesses 2 campos, e `env.plan_runtime` só em `LN_DEFER`/`FIDESLIB_BTS_EVALMOD`. `export_env` exporta `LN_DEFER=1` antes de `model.block_weights` ler o ambiente (leitura preguiçosa): os 12 blocos do gêmeo saem com `ln_defer=True`. `impl/norm.py` e `weights.py` não leem o ambiente. **Ressalva:** `block.py:15` lê `LN_DEFER_EXIT` no import, e `model.py`/`block.py` são importados ANTES do `export_env` (espião: `block_imported_before_export=True`). Inofensivo aqui: o runtime não define a chave, o default é 3 e é o da captura "e3". `cargo test -p pc-config`: 4 passam, `shipped_stamps_round_trip` falha em `gpt2_prefill128_n32a4_pfbh_cm` (`PLAN_PCMM_PREDIV_CAP`), **igual em d97a6f7** (pré-existente). Como esse teste para no 1º arquivo, rodei no clone uma variante que percorre os 1456 arquivos: só os 4 planos de prefill falham, e o PLAN_CMD e os placements da entrada passam. |
| 4 | Plano 108/token; 0 erros; 0 fora da rota; PLAN_CMD; carimbo; time_model | **CONFIRMADA (com ressalva)** | `compile_check` 14 estágios: 0 erros, por bloco 7,8,8,8,8,8,7,7,8,7,8,10,1 + 13 = 95 + 13 = **108** (recorde 114). `route_mix`: 0 deliberados fora da rota do recorde; 6 SparseB-512 a menos. `time_model`: 1429,9 contra 1344,3 ms, reproduzido (`plan_checks.log`). PLAN_CMD: runtime com `CHAIN_PRESET=n32a4 FIDESLIB_BTS_EVALMOD=d27 LN_DEFER=1`; receita com BTS_LEVEL 32/26, `512:24`, `26:24`, `32:1`; configs kc16r_cm. Carimbo `config` dos 15 JSONs: idêntico ao do recorde, sem contradição com o runtime, mas **sem** `bts.evalmod`/`model.ln_defer` (ver C3). Estágio 13 replanejado pelo ILP (raise drops diferentes do recorde). Blocos 12 e feedback saíram do min-cut, não do ILP (ver C5). |
| 5 | `prepare()` do servidor × grafo `lnd_on_e3m` | **CONFIRMADA (reproduzida na CPU)** | `prepare_entry.py` replica `server.py decode()` até o `manifest()`: mesma ordem de import, `Gpt2Handler`, `PlannedRun.prepare`, `Session.key_manifest`. Modo de baixa memória: pesos só do bloco corrente e valores com 1 slot (o gêmeo é só estrutura). Validado pelo controle do item 1. Resultado na entrada: compile 0 erros, 95 + 13, 124 chaves idênticas às do recorde (só o `contract` muda), entrada do bloco 0 no nível 24. As 16 posições passam na checagem estrutural do `precompute` (`diff_block` + passos dinâmicos), na entrada e no recorde (`full16_*.log`). **Achado:** o fluxo gravado pelo servidor NÃO é igual ao `lnd_on_e3m` (ver C1): mesmos `v_N` e mesmas ops, mas outro número de nós `auto_bootstrap` (bloco 12: 4 contra 1026) e outras flags `fired` dos hints. A causa foi isolada: sem o `FIDESLIB_BTS_EVALMOD` no gêmeo, o fluxo fica idêntico à captura. Os programas compilados dos dois grafos são equivalentes (mesmas ops, mesmos sítios), exceto pelo nível de entrada do bloco 0 e por nomes de plaintexts. |
| 6 | Regras, check_pr, segurança | **CONFIRMADA (check_pr vazio)** | Nenhum caminho protegido mudou (harness, runner, prompts, records, README, CLAUDE.md, .github, LICENSE, CLA, pc-client, python/client, params.rs); em `speedrun/` só `server.py`. `check_pr.py --base origin/speedrun --head HEAD` sai com 0, **mas é vácuo**: sem pasta de record ele retorna antes das checagens de campos e CKKS. Testei as funções puras com o bloco `ckks` do recorde 4 (o da entrada): `security` dá log QP 1734 ≤ 1747, `fields` = [], `build_safety` limpo. `params_hash` do pedido de chaves idêntico ao do recorde, logo cadeia e segredo inalterados. |
| 7 | Bundle aplica num clone limpo; cargo check | **CONFIRMADA** | Clone novo em `sanity2/clone`, só com `origin/speedrun` = d97a6f7 (raso, porque `work/perseus` é raso), sem o commit da entrada. `git bundle verify`: "is okay". `fetch` do bundle + `switch`: HEAD = **44858c8**. `cargo check --offline --workspace --release` passa, só com 2 avisos pré-existentes em pc-kernels (`cargo_check_clone.log`). pc-shadow: todos os testes passam. pc-core: as únicas falhas são os testes de oráculo que exigem o dump privado `/data02/...`. |
| 8 | Revisão do `gpu/run_gpu.sh` | **BUGS ENCONTRADOS** | SHAs, backend `perseus-lean`, CLIENT/WEIGHTS, worktrees e alternância R,E,E,R,... estão certos. Confirmados por simulação: B1 (falha nunca detectada), B2 (score aborta o script e pula o prefill), B3 (setup não re-roda). Lista abaixo. |

## Bugs e riscos, por gravidade

### Alta (corrigir antes de alugar)

- **B1 `gpu/run_gpu.sh:65-73,85`: falha de run nunca é detectada.** Bash desliga o `errexit` dentro de `$(...)`,
  então `run_one` ignora o código de saída do `entry.sh` e sempre ecoa um caminho. Quando a falha vem antes do
  run.py criar o diretório, o caminho é `/run.json`. O `|| { log FAILED; ...; return 1; }` da linha 85 é código
  morto (simulado em `simgpu/sim.sh`). Cenário: o servidor da entrada morre no `prepare()`; o script roda os 12
  runs (~1 h de GPU) e só o `score.py` reclama no fim ("refused").
  Correção: `... >> "$OUT/${track}_$who.log" 2>&1 || return 1`, depois `[ -n "$after" ] || return 1`, e checar
  `void`/`error` no run.json (`python3 -c ...`).
- **B2 `gpu/run_gpu.sh:90-92,99-100`: `score.py | tee` sob `set -euo pipefail` aborta o script.** Saída 1 (não é
  record, ou falha no KL) ou 2 (recusado) mata o script. Com `all`, se o decode não bater o recorde, **o prefill
  nunca roda**. E `track prefill 1` sempre dá "refused: pair at least 2 of each" (saída 2, simulado em
  `simgpu/sim_score.sh`), sem nem imprimir a qualidade, porque a velocidade levanta erro dentro do mesmo `try`.
  Correção: `score.py ... | tee ... || true` (guardar o código à parte); no prefill, rodar `score.py --entry`
  sem `--record` (só qualidade) ou 2 pares.
- **B3 `gpu/run_gpu.sh:35`: `setup` não pode ser re-rodado.** O `git fetch BUNDLE ln-defer:ln-defer` com `ln-defer`
  em checkout dá "refusing to fetch into branch ... checked out" (saída 128, testado no clone). Depois de qualquer
  falha, `all` morre no início. Também não se confere que HEAD = 44858c8 (a linha 36 só loga).
  Correção: `git fetch "$BUNDLE" ln-defer && git checkout -B ln-defer FETCH_HEAD` e
  `[ "$(git rev-parse HEAD)" = 44858c8384e05b3386ef9a9b5fb15bc8dd5fb537 ]`.

### Média

- **R1 (só GPU): a série d27 nunca foi executada em GPU nenhuma.** Na rota dense ela troca o PS de nível único
  (m = 2, k = 5) pelo PS geral recursivo (m = 3, k = 4; `ps_shapes.log`), combinação inédita na dense. Na
  SparseB, a K24 d27 tem 28,2 bits contra 35,2. O modelo de banda e de CF do planner (`plan/btserr.py`, tabelas de
  rota) foi medido com as tabelas shipped e não conhece o EvalMod. O passo `bts` do script é o 1º teste, mas só da
  dense; a SparseB d27 só aparece no KL do harness. Rode `bts` e um único run da entrada ANTES dos 6 pares.
- **C1 `python/perseus/impl/fakecap.py:145` + `crates/pc-py/src/shadow.rs:64`: o gêmeo do servidor grava com os
  pousos d27, mas o `lnd_on_e3m` foi capturado com os shipped** (o `cap.sh` não define `FIDESLIB_BTS_EVALMOD`).
  Medido: o compilador apaga os `auto_bootstrap` e o plano fixa os hints, então os programas saem equivalentes.
  O que muda de fato: (a) a entrada do bloco 0 cai do nível 26 (planejado) para 24. O cliente cifra em 24, o 1º
  segmento do bloco 0 roda com 2 primos a mais e o SPRU `v_89` recebe no nível 42, não 44. É viável (0 erros) e
  custa pouco; (b) a linha `graph=` do PLAN_CMD não é literalmente o que o servidor compila. Sem o switch no gêmeo,
  o compile falha (113 `plan_level_error` por bloco: o nível de K/V que o plano espera é o do d27), então o desenho
  está certo. Sugestão: recapturar com `FIDESLIB_BTS_EVALMOD=d27`, ou documentar.
- **B4 `gpu/run_gpu.sh:69-70`: todos os 12 runs usam 1 request no prompt 0** (`--requests 1 --offset 0`, defaults).
  O teste t do KL fica sobre 6 amostras do MESMO prompt dev: evidência fraca de KL para o conjunto oculto.
  Sugestão: `--requests 4` ou `--offset 4*(i-1)` por par (os dois do par no mesmo offset).
- **B5 `gpu/run_gpu.sh:26-27,30`: as checagens do ambiente são superficiais.** `command -v bwrap` não prova que o
  bwrap funciona (Ubuntu 24.04 bloqueia userns via AppArmor). Containers (RunPod etc.) não têm systemd de usuário.
  Tudo isso falha depois de ~1 h de builds. Sugestão: smoke `bwrap --ro-bind / / --dev /dev true` e
  `systemd-run --user --scope true`, ou um run do `dummy_server` do harness, antes dos builds.

### Baixa

- **B6 `gpu/run_gpu.sh:19,30`:** se o `nvcc` não estiver no PATH, `CUDA_HOME` cai para `/usr/local/cuda`, mas o PATH
  não é estendido e a linha 30 aborta ("nvcc missing"). O `build.rs` usa `$CUDA_HOME/bin/nvcc` e `$CUDA_HOME/lib64`.
  Com `/usr/bin/nvcc` de distro, `CUDA_HOME=/usr` dá `lib64` errado. Correção: `export PATH=$CUDA_HOME/bin:$PATH`
  e testar `[ -x "$CUDA_HOME/bin/nvcc" ]`.
- **B7 `gpu/run_gpu.sh:57-61`:** `[ -x "$B" ] ||` não recompila um pc-bench velho. O pc-bench só conhece `n32`/`n64`
  (`pc-bench.rs:205`): mede a cadeia n32, não a n32a4 servida. Só a rota dense.
- **B8 `gpu/run_gpu.sh:79-89`:** o `nvidia-smi` em segundo plano fica vivo se o script morrer (sem `trap ... EXIT`).
  O pico da linha 92 mistura recorde e entrada (o run.json já tem `peak_gpu_mib` por run).
- **B9 `gpu/run_gpu.sh:40-44`:** 4 builds release com `PC_NVCC_JOBS=2` (default do `pc-kernels/build.rs`) no tempo
  pago da GPU. Exportar `PC_NVCC_JOBS=$(nproc)`. Também falta checar, depois do build, que o cliente lean continua
  sem mudanças rastreadas (o `run.py` recusa "has changes to its tracked files").
- **C2 `python/perseus/examples/gpt2/block.py:15`:** `LN_DEFER_EXIT` é lido no import, antes do `export_env`
  (`serve/gpt2_decode.py:42` importa o model antes da `:46`). Um plano que ponha `LN_DEFER_EXIT` no runtime= seria
  ignorado pelo servidor. Correção: ler `os.environ` dentro de `transformer_block`.
- **C3 carimbo `config` dos `block_*_placement.json` + `python/perseus/plan/contract.py:20`:** o carimbo vem do
  `capture_config.json`, anterior aos campos novos, e `LOAD_BEARING_ENV` não tem `LN_DEFER` nem
  `FIDESLIB_BTS_EVALMOD`. O contrato não pega uma sessão sem os switches. Mitigado: testei, e sem `LN_DEFER` no
  gêmeo o prepare falha alto (603 erros de compile, `prep_nolnd.log`); sem evalmod também (113 por bloco).
- **C4 `python/tests/test_fake_shadow.py:32`:** asserta 260 campos; são 262 desde 7e497ca. O teste falha; o CI só
  roda o `check_pr`.
- **C5 `bootstrap_placements/lnd_fin_e3_em9_cm/PLAN_CMD.txt:2,8,10`:** `tool=` diz `PLACER=mincut`; `solver=`
  ("every block the ILP ... proven optimal", "Block 12 ... 596 s") e `capture=` foram copiados do recorde. Os blocos
  12 e feedback são min-cut, e a captura real é `cap.sh` com `LN_DEFER=1` + merge. Corrigir para as Disclosures do
  PR.
- **C6 pc-check sem o switch:** `crates/pc-tools/src/check_ccmm.rs:696,705,775`, `check_pcmm.rs:595,678,792`,
  `check_bts.rs:269,448,457`, `replay_g0.rs:563,576` ainda usam `n32_shipped()` sem `set_evalmod`. Com
  `FIDESLIB_BTS_EVALMOD=d27`, testam as tabelas shipped sem avisar. Só ferramentas.
- **C7 `bootstrap_placements/lnd_fin_e3_d27sb_cm/PLAN_CMD.txt` (untracked, não servido):** o runtime= não tem
  `LN_DEFER=1`.
- **Info (upstream):** `fakecap.py:100-101` (`self.vec = self.vec[:1]` é uma vista e segura o buffer cheio) e o
  armazém de slot pts fazem o gêmeo passar de 6 GB (por isso o OOM da auditoria anterior). No servidor (160 GB) não
  importa.
- **R2:** o ganho (−2,8 a −4,1 % do token) é modelo (route table do recorde 3 × razões, + 12-36 ms do EvalMod dense
  d27). Não inclui o efeito, pequeno, da entrada do bloco 0 em 24. A regra 5 exige ≥ 1 % e 3 erros-padrão.

## Arquivos criados (todos em `~/pesquisa/perseus-speedrun/subagent/sanity2/`, total ~2,1 GB)

- Scripts: `env.sh`, `prepare_entry.py` (CPU do `server.py decode()` + comparação de grafos + `FULL16`),
  `mem_probe.py`, `mem_trace.py`, `cfg_check.py`, `compare_progs.py`, `ptref_diff.py`, `graph_diff.py`,
  `keyneeds_modes.py`, `simgpu/sim.sh`, `simgpu/sim_score.sh`.
- Logs: `prep_{entry,record,entry_noem,nolnd}.log`, `full16_{entry,record}.log`, `cfg_check.log`, `plan_checks.log`,
  `compare_progs.log`, `ptref_diff.log`, `graph_diff.log`, `keyneeds_modes.log`, `d27sb_compile.log`,
  `pcconfig_test_{head,d97a6f7,44858c8}.log`, `pcconfig_allstamps_44858c8.log`, `cargo_test_shadow_core.log`,
  `cargo_check_clone.log`, `check_pr.log`, `ps_shapes.log`, `pytest_fields.log`, `mem_probe.log`, `mem_trace.log`,
  `code.diff`.
- Diretórios: `prep_*`, `full16_*` (grafos gravados pelo gêmeo e keyspecs, ~16 MB cada), `progs_cmp/` (programas
  salvos), `simgpu/` (runs falsos), `tmp/` (TMPDIR), **`clone/` (2,0 GB: clone + `target/`)**. No clone há 2
  arquivos untracked meus: `crates/pc-config/tests/sanity2_roundtrip.rs`, `crates/pc-shadow/examples/sanity2_ps.rs`.

Para apagar depois (não apaguei nada): `subagent/sanity2/clone/` (o maior) e, se quiser, o resto de `sanity2/`. Fora
de `sanity2/`, ficaram 5 bytecodes que eu gerei (ignorados pelo git) em `work/perseus-r4`:
`speedrun/__pycache__/record1.cpython-312.pyc`, `speedrun/harness/__pycache__/check_pr.cpython-314.pyc`,
`python/perseus/client/__pycache__/__init__.cpython-312.pyc`, `python/perseus/serve/__pycache__/__init__.cpython-312.pyc`,
`python/perseus/serve/__pycache__/gpt2_decode.cpython-312.pyc`. Nenhuma captura `graphs/sanity2_*` foi criada.

## Como reproduzir o teste do `prepare()` (o achado principal)

```bash
cd ~/pesquisa/perseus-speedrun/work/perseus-r4 && source ~/pesquisa/perseus-speedrun/subagent/sanity2/env.sh
export OMP_NUM_THREADS=4 PERSEUS_SLOTVEC_CACHE=0 LOWMEM=1 FULL16=1 && source scripts/local_env.sh
export PYTHONPATH=$R4/python:$R4
systemd-run --user --scope -q -p MemoryMax=6G $PYTHON $S2/prepare_entry.py bootstrap_placements/lnd_fin_e3_em9_cm \
  $S2/full16_entry graphs/lnd_on_e3m graphs/gpt2_decode_python_n32_kc16r_cm      # ~25 s, pico 3,8 GB
# controle: o mesmo com bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm e graphs/gpt2_decode_python_n32_kc16rm
# causa do C1: TWIN_NO_EVALMOD=1 (fluxo = captura, compile falha); TWIN_NO_LNDEFER=1 (compile falha alto)
```

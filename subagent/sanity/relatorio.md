# Sanity check: plano Decode-8 `fin_rc3b_m_cm` + commit `d8a7643` (2026-10-28)

Auditor cético e independente. Tudo abaixo foi lido no código ou executado na CPU nesta sessão, sempre sob
`systemd-run --user --scope -p MemoryMax=4G` (2G em duas checagens leves). Nada foi editado em `work/perseus`,
`work/perseus-rec`, `scripts/`, `configs/`, `plans/`, `README.md` nem `HANDOFF.md`. O único efeito fora desta pasta
são artefatos de build em `work/perseus/target`, criados por `cargo test`.

Scripts e saídas ficam em `subagent/sanity/`:
- `sites.py` / `sites.out`: compila os 14 estágios encadeados e lista erros, refreshes por estágio e deliberados;
- `delib_cmp.py` / `.out`: compara cada refresh deliberado com o do recorde (tupla de rota inteira + pouso compilado);
- `emscan/` / `emscan.out`: crate próprio sobre `pc-core` + `pc-shadow`; mede os níveis do EvalMod das duas séries;
- `manifest_check.py` / `manifest.out` / `spec_*.json`: compila com `config=` da sessão e roda `Session.key_manifest`;
- `cfg_check.py`: Config da sessão (`plan_runtime`) e pousos do fakecap;
- `cfg_rebuild.py` / `.out`: `build_cfg.build()` em memória, sem escrever nada, e checagens estruturais da softmax;
- `cfgdiff.py`: diff por campo entre configs;
- `band_spot.py` / `.out`: spot check das faixas no held-out;
- `prepare_cpu.py` / `prepare_rc3b.log`: tentativa do `prepare()` completo (morreu por OOM);
- `plans/`: cópias de plano com o switch ligado ou desligado, só para compilar.

## Vereditos

| # | alegação | veredito |
|---|---|---|
| 1 | switch desligado = recorde; `d27` = 2 primos mais raso em toda rota menos a SPRU; toda rota de GPU com tabela d27 + 4 DA | **CONFIRMADO** |
| 2 | `fin_rc3b_m_cm`: 98 + 12 = 110; 0 erros; com erro sem o switch; 0 deliberados fora da rota; mix 36/25/37 | **CONFIRMADO** |
| 3 | remap aplicado 2× ⇒ `24:14,26:24` manda o push K/V para a SPRU; `26:24` sozinho é seguro; SPRU = lixo | **CONFIRMADO** (só a checagem na GPU fica NÃO VERIFICÁVEL) |
| 4 | faixas da rodada 0: sem dev nem held-out na calibração, regra do recorde, sem l2=0, held-out 0 fora, folga 2,79/5,31 | **CONFIRMADO** (as 16k janelas não foram rodadas de novo; o log bate) |
| 5 | `build_cfg` reproduz `cand_dfr13_cm` bit a bit; `cand_rc3` só muda softmax, GELU, defer 1,3 e cutmax do kc16_cm | **CONFIRMADO** |
| 6 | algo que derrube o `prepare()` / harness | nenhum bloqueador no código; **3 pendências de entrega** (abaixo); `prepare()` completo NÃO VERIFICÁVEL (OOM a 4G) |

## 1. Switch `bts.evalmod`

**Tabelas.** `git diff a3dff85 d8a7643 -- crates/pc-core/src/bts_setup.rs` só ADICIONA linhas:
`EVALMOD_CHEB_D27`, `SPARSE_B_CHEB_K24_D27`, `D27_DA_ITS` e `enum EvalMod` (`bts_setup.rs:141-270`).
`EVALMOD_CHEB`, `SPARSE_B_CHEB_K24`, `EVALMOD_DA_ITS` e `SPARSE_B_DA_ITS` ficaram intocados.

**Sem switch, nada muda.** Com `Shipped` o deslocamento é 0 (`route.rs:186-188`). Em cada ponto:
- compilador: `compile/mod.rs:243-244`;
- Shadow do pc-py: `shadow.rs:107-108`;
- fakecap: `shadow.rs:55-66`, `fakecap.py:145`. Para o recorde dá `{0:(32,2),1:(14,2),256:(22,2),512:(24,2)}`, o
  mesmo que o `parse_landings` antigo;
- GPU: `set_evalmod(Shipped)` devolve as mesmas tabelas e 5 DA (`bts/mod.rs:174`, `sparse_classic.rs:91`,
  `sparse_b.rs:55-63`).

| medição | resultado |
|---|---|
| compile do recorde (sem switch) | 0 erros |
| compile do recorde (com d27) | 34.085 erros (o compilador vê os pousos novos) |
| compile do Prefill-128 `pfbhrd_cm` | 0 erros |
| `cargo test -p pc-shadow --lib` (`multi_stage_landings`, `evalmod_series_levels` e os demais) | 4/4 |
| `cargo test -p pc-config` | 4/5; `shipped_stamps_round_trip` cai em `fin_dfr_m3_cm` (`PLAN_B12_ENV_CAP`), como o README diz |

**Sítios da GPU** (`grep` de `BtsSetup::n32_shipped|SparseBCfg::n32|ClassicCfg::n32|evalmod_coeffs()|*_DA_ITS|ps_k24`
no código de runtime). Os únicos construtores de runtime ficam em `pc-py/src/session.rs`:

| rota | onde | o que recebe |
|---|---|---|
| dense / StC-first | `:651-652` | `set_evalmod` |
| clássica do plano (plain) | `:685-686` | `set_evalmod` |
| SparseB | `:699-704` | `c.evalmod = em`; `em_series()` usa `dense()` no caso K = 16 e `sparse_k24()` no K = 24, sempre com `da_its()` |
| clássicas esparsas | `:713` | `set_evalmod` |
| `high_shapes` (CCMM) | `matmul.rs:86-87` | `set_evalmod` |

O `SparseBPrecomp` é montado a partir desses cfgs (`session.rs:1190`). Os outros usos de `evalmod_coeffs()`,
`ps_k24()` e `PsPlan::evalmod()` são testes (`evalmod.rs:1244,1268`, `pc-shadow evalmod.rs:881`).

**Números fixos restantes:**
- `fake_bts_out` 34 (modo "fake");
- `_N32_LAND = 26` do PCMM (só com `PF_LINEAR=pcmm`);
- `PF_CCMM_FAKE_LAND` 28.

Nenhum está no caminho Decode-8.

**Níveis por série.** Medi `build_evalmod` (o mesmo que `EvalModPlan::program` da GPU chama, `bts/evalmod.rs:359-360`)
para todo topo de entrada ímpar de 25 a 53, nl 1 e 2, K16 e K24. A d27 consome sempre exatamente 2 primos a menos
que a shipped, com o mesmo nl de saída: `mismatches: 0` (`emscan.out`). Logo o `out_meta` real da GPU pousa 2 primos
mais raso em toda rota com EvalMod, como o compilador assume.

**Pontos frágeis** (não são bugs no plano atual):
- `shift_evalmod` só desloca os valores padrão. Os da receita do plano (`BTS_LEVEL=26`, `PLAN_STC_FIRST_FALLBACK=32:1`,
  `PLAN_SPARSE_BTS_OUT=512:24`) têm de vir já em d27, e vêm. Um valor errado aqui só o compile_check pega.
- `quirks.fakecap_landings` é SEMPRE deslocado (`shadow.rs:62-66`), então tem de ser escrito na unidade shipped.
- O modelo de acurácia do planner (`plan/btserr.py`) foi medido com as tabelas shipped. A K24 d27 tem 28,2 bits
  contra 35,2. Isso só a GPU confirma.

## 2. Plano `fin_rc3b_m_cm`

O `PLAN_CMD.txt` tem `FIDESLIB_BTS_EVALMOD=d27` na linha `runtime=` e `PLAN_DELIB_LANDING_REMAP=26:24` na receita.
Contra o recorde, a receita só muda `BTS_LEVEL` 34→32 e 28→26, `512:26`→`512:24`, `PLAN_STC_FIRST_FALLBACK` 34→32
e o remap.

| combinação (14 estágios, `sites.py`) | erros | blocos 0-12 / estágio 13 |
|---|---|---|
| `graphs_combo/rc3b` + `fin_rc3b_m_cm` | **0** | **98** / 12 |
| idem, sem o switch | 33.320 | — |
| recorde | 0 | 108 / 12 |

`route_mix.py` dá 0 fora da rota; recorde 36/25/47, plano **36/25/37**.

`route_mix.py` só compara `(slots, real)` e casa os sítios pelo nome da variável. Por isso refiz a comparação com a
tupla inteira e o pouso (`delib_cmp.out`):
- os 12 pushes K/V são dense, StC-first, real, drop 0, iguais aos do recorde, e pousam raw **24** (recorde 26);
- os 6 deliberados do CutMax são SPRU e pousam raw 14, como no recorde.

## 3. Remap aplicado duas vezes

**Mecanismo.** O mesmo grafo passa duas vezes pelo remap, no Python e no Rust:

| lado | 1ª aplicação | 2ª aplicação |
|---|---|---|
| planner (Python) | `Graph.from_nodes` aplica o remap (`ir.py:120-145`) | `planner.py:646-648` chama `erase_reactive_bootstraps`, que remonta via `Graph.from_nodes` (`place.py:113`) |
| compilador (Rust) | `from_raw` (`graph.rs:212-215`) | `erase_reactive_bootstraps` (`graph.rs:407-411`, "from_nodes remaps it again") |

Com `24:14,26:24` acontece `26→24→14`. A rota do deliberado sai do pouso capturado (`apply_plan.rs:88-100`,
`route_for_landing` em `compile/mod.rs:175`). Ninguém olha a tag: "the tag is advisory" (`check.rs:163`).

**Medido.**
- `fin_dfr13_m_cm` + d27: **0 erros**, com os 12 deliberados na rota 1 (SPRU).
- O mesmo plano com só `26:24`: os deliberados vão para dense e o plano desmonta (33.179 erros). Isso prova que o
  "98" antigo dependia da SPRU.

**`26:24` sozinho é seguro.**
- Os grafos `rc3b`, `kc16_cm` e `fin_dfr13` só capturam deliberados em 26 (os 12 K/V) e 14 (CutMax).
- Em d27 os pousos raw são: dense StC-first 24, ordem shipped 30, 512 com drop 1 em 22, 256 em 20, SPRU 14.
  Só a dense pousa em 24, e 24 não é chave do remap, então a 2ª passada não muda nada.
- O gêmeo do `prepare()` com fakecap d27 grava 30 − 6 = 24, e o remap também não mexe nisso.

**SPRU num push K/V dá lixo: correto.** Pelo próprio código:
- `spru.rs:6`: "The output is the slot mean (a 1-slot fold) in every slot";
- `fold.rs:5-6`: um bootstrap esparso de valor não periódico devolve a média da classe.

O compilador não recusa nada disso (0 erros acima). Não achei checagem de periodicidade na GPU, mas sem GPU isso fica
NÃO VERIFICÁVEL.

## 4. Faixas da softmax (`configs/cand_rc3`)

**A faixa é a da config.** O `band_from_seed(init_alpha, init_beta)` de cada camada bate com o `sm_split_rc3_t0.01.json`.
Nenhuma camada tem l2 = 0 (l2 = 3,2,1,2,4,2,2,2,1,1,2,3). Em todas, o topo ≥ 1,05·exp((clip_hi−mid)/2^l2) e o divisor
das cabeças com padding, exp(−mid/2^l2), fica dentro da faixa.

**Consistência da config:**
- n_squarings / refinement_iters / delta1 / delta2 / listas por rodada batem;
- os `recip_iters_cheb` são exatamente o `smcheb.design` refeito;
- o domínio da exp (`cheb_a/sf`) é igual ao meio-intervalo do clip e ao do recorde (`cfg_rebuild.out`).

**A fórmula do runtime bate com a do `band_check`.** No caminho cifrado:
- `_thor_exp` (`attention.py:619-635`): exp Chebyshev em (s − mid), l1 quadrados, máscara ativa 0,5/kc só nas kc chaves;
- rodada 0 de `smcheb.rounds` (`smcheb.py:93-109`): seed = `init_alpha`/`init_beta` com scale 1,0 sobre a soma do
  head, isto é, d = média_k exp((s − mid)/2^l2);
- o gêmeo `ref.py:85-101` faz o mesmo, e é isso que o `owt_windows.divisors` calcula.

**Ressalva medida.** O `band_check` usa scores do GPT-2 exato; o runtime vê scores do modelo aproximado. Rodei
`band_spot.py` em 261 janelas: 128 inícios de documento do held-out, 128 aleatórias do held-out e as 5 de estresse.
Os divisores do gêmeo (exp polinomial sobre os scores do `forward_approx`) e os exatos diferem menos de 0,5 %:

| conjunto | janelas fora | folga mín. embaixo | folga mín. em cima |
|---|---|---|---|
| spot check (261 janelas) | 0 | 3,05 | 6,31 |
| log `work/band_holdout.txt` (17.282 janelas) | 0 | 2,79 | 5,31 |

O log do held-out completo também reproduz recorde 2,57/5,79 e antigo com 17 fora.

**Proveniência:**
- calib = [0, 1M), 865 inícios + aleatórios; held-out = [1M, 2M), 898 + 16.384 = 17.282; interseção 0
  (maior offset de calib 999.911, menor de held-out 1.000.059);
- o cache `round0_d_calib.json` (16.384 janelas, 865 inícios) é posterior ao `sm_split.py` atual;
- dev e teste são documentos C4 (`runner/make_prompts.py:1-8`), sem sobreposição com o pool OWT.

Que nenhuma escolha tenha olhado o held-out é processo e não dá para verificar; o código não o lê.

## 5. Configs

`build_cfg.build()` em memória:
- split `m4h6_t0.01` → **bytes idênticos** a `cand_dfr13_cm` (767.820 B);
- split `rc3_t0.01` → idênticos a `cand_rc3` (770.627 B).

As 3 cópias de `cand_rc3` são iguais: dossiê, `work/perseus` e `work/perseus-rec`.

Contra `gpt2_base_n32_kc16_cm`, só mudam:
- `softmax` em todas as camadas (split, faixas, série exp);
- `defer_norm` só em h1 e h3 (h11 sem);
- `softgelu.cheb_coeffs` em 8 camadas (0,2,4,5,6,8,9,11).

`norm`, `cutmax` e `model` são idênticos.

## 6. `prepare()` / harness

**OK:**
- todos os 44 nomes da receita e os 28 da linha `runtime=` são campos conhecidos;
- a Config da sessão (`gpt2_decode._config`) tem `bts.evalmod='d27'`; a do recorde e a do Prefill têm `None`;
- `Program.compile(config=sessão)` dá 0 erros, então a checagem carimbo × sessão passa, inclusive no estágio 13 com o
  carimbo do recorde;
- `Session.key_manifest` (sem device) monta os planos de host StC-first, dense, SparseB e clássica com d27. Saem
  124 chaves, e o pedido é **idêntico ao do recorde, menos o `contract`** (`c530fc80b818f119` contra
  `4069713ff8a49620`). Isso é inofensivo: o próprio `server.py:82` calcula o contrato do hello, e o cliente do harness
  não o confere (`perseus_core_client.py:35-70`);
- o push K/V compilado em raw 24 é o que a GPU entrega (§1).

**NÃO VERIFICÁVEL:** o `prepare()` completo (`Gpt2Handler.manifest`, gravação do gêmeo de CPU) morreu por OOM a 4G
três vezes. Precisa de mais RAM; na máquina do servidor deve passar.

**Pendências de entrega** (não são erros das alegações, mas derrubam a rodada se esquecidas):
1. `speedrun/server.py:34`: `DECODE_PLAN` ainda é o recorde, e `runner/entry.sh:25` chama `serve.sh` sem `--plan`.
   Sem mudar isso, o harness serve o recorde.
2. `bootstrap_placements/fin_rc3b_m_cm/` e `configs/model/approximation/cand_rc3/` estão **untracked**: não estão no
   `d8a7643` nem no bundle. Os grafos não são necessários no servidor (o `prepare()` grava os próprios).
3. Falta `block_0_feedback_placement.json`. Só o `run_generate.py` exige; o harness (14 labels, sem variante) não.

**Observação.** Para o crate `emscan`, o `cargo` atualizou o índice do crates.io (download somente leitura). Não houve
nenhum outro acesso à rede nem a servidor.

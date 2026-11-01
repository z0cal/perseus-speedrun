# Ganhos rápidos adicionais: Decode-8 (subagente, 2026-10-27)

Base de comparação: `split4h6t2` (em9 + divisão l1/l2 + alvo 1e-2). As contagens abaixo são **proxy min-cut, blocos 0-12,
receita em9**, todas da mesma pipeline (`capture_rec.sh` + `plan_count.sh`).

- **Re-medi a base:** o proxy de `split4h6t2` dá **96**, não 99 (o 99 da tabela é o `split4h6t3`). Todos os deltas abaixo
  são contra 96. A ILP desse candidato é 94.
- **O proxy é ruidoso.** Ele não é monótono: há ±1-2 de ruído entre combinações equivalentes. Por isso as conclusões
  firmes estão na coluna "por quê"; a ILP decide.
- **Garantia para as mudanças de config (itens 3 e 4):** elas só encurtam caminhos do grafo, então a ILP não pode piorar
  com elas.

## Medições (KL em 8 janelas OWT, `kl_eval.py`; proxy por `subagent/search2.py`/`plan_count2.sh`)

| candidato | o que muda | KL média (max) | proxy 0-12 | rota (dense/512/1) | ~ms bts* |
|---|---|---|---|---|---|
| `sa_s6t2b` (base) | split4h6t2 | 2,67e-6 (5,7e-5) | **96** | 40/31/25 | 2130 |
| `sa_s6t2r0` | exp minimax **relativo**, r=1 (−12 níveis/token) | 2,68e-6 | 96 | – | – |
| `sa_s6t2r2.0` | idem, r=2 onde a≤2,0 (−20 níveis) | 2,66e-6 | 95 | – | – |
| `sa_s6t2r3` | idem, r=2 em todas (−24 níveis) | 3,90e-6 (5,1e-5) | 95 | 40/30/25 | 2111 |
| `sa_s6t2gb` | GELU 62→60, 254→248 (−8 níveis) | 4,07e-6 (9,5e-5) | 95 | 39/30/26 | 2098 |
| `sa_s6t2r3gb` | r3 + GELU | 5,07e-6 (7,8e-5); **32 jan.: 4,94e-6** | 94 | 39/29/26 | 2080 |
| `sa_s6t2r0gb` | r0 + GELU (versão segura) | 4,07e-6 | 95 | – | – |
| `sa_s6t2_defer` | normalização adiada, 12 camadas | = base (mesma conta) | 95 | 41/29/25 | 2121 |
| `sa_s6t2_deferA` | adiada em {1,3,6,11} | = base | **92** | 38/29/25 | 2035 |
| `sa_s6t2_deferB` | adiada em {3,6,11} | = base | 92 | 37/30/25 | 2025 |
| `sa_r3gb_d1_3_11` | r3 + GELU + adiada {1,3,11} | 5,07e-6 | **91** | 37/28/26 | 2004 |
| `sa_r0gb_d1_3_11` | r0 + GELU + adiada {1,3,11} (segura) | 4,07e-6 | **92** | – | – |
| `sa_s6t2b_sp22only` | SparseB pousa +1 nível (EvalMod de 8 níveis só no SparseB) | – | **93** | 39/29/25 | 2064 |
| `sa_s6t2b_kv22` | só o push K/V +1 nível (controle) | – | 97 | – | – |
| `sa_s6t2_deferA` + SparseB +1 | | = base | 91 | – | – |
| `sa_r3gb_d1_3_11` + SparseB +1 | | | 91 | – | – |
| `sa_s6t2_nokv` | sem o refresh deliberado do push K/V | = base | 101 (pior) | 36/40/25 | 2182 |
| `sa_s6t2w2` | exp intervalo 4×, minimax **absoluto** grau 14 | **3,1e-4** (4,9e-3) | – | – | – |
| `sa_s6t2gb94` | GELU 1/3/10 grau 94→60 | **1,8e-4** (1,6e-3) | – | – | – |

\*Latência só dos bootstraps com a tabela `bts_ms` padrão do planner (dense 28,77, 512 18,59, SPRU 16,10 ms; GPU antiga,
só relativo). Linhas com "–" não foram medidas.

## Ranking

### 1. Normalização adiada da última rodada da softmax ("normalize after PV")

- **Mecanismo.** Hoje a última rodada calcula `y = z·r` (r = 1/s por cabeça, Goldschmidt) e depois
  `softmax_v = Re(Σ V·(y − iy))` → `tok0.h` mask. Como r é tH-periódico e constante nos t slots de cada cabeça, ele comuta
  com o rotsum de tokens, com o `im_cleanse` e com a máscara de slots. Fica:
  `w = mask(Re(Σ V·(z − iz)))` no ramo raso de z, em paralelo ao Goldschmidt, e depois `w·r`.
  - O caminho crítico perde o produto z·r e a máscara: **−2 níveis por camada**, com exatamente a mesma conta.
  - É o truque do FlashAttention / "online softmax": THOR e NEXUS dividem antes do PV.
- **Onde.**
  - `python/perseus/impl/attention.py::_softmax_recip` (devolver (z, r) na última rodada).
  - `complex_softmax_v`, ramo `per == 1` (SM_PERIODIC=1, que vale para ≤ 16 chaves).
  - Atrás de um switch de runtime com lista de camadas.
  - `ref.py` não muda (a conta é idêntica).
  - Protótipo: `subagent/patch/defer_norm.py` (monkeypatch só para a captura).
- **Medido.** Proxy 96 → **92** com as camadas {1,3,6,11}.
  - Em todas as 12 camadas dá 95: as camadas 2, 8, 9 e 10 pioram 1 cada (l2 ≤ 1, ramo z fica vivo mais tempo). A
    escolha tem que ser por camada.
  - Com o item 3 + item 4: **91**.
  - A captura fake passa (top1 ok, KL do oráculo 8,8e-7, igual à base).
- **Risco.** KL: nenhum em aritmética exata; o ruído CKKS só muda de ordem. Regras: ok. Revisão: código claro, sem custo
  de legibilidade.
- **Esforço.** 3-5 h de Python + teste CPU, recaptura e ILP. O ganho na ILP com escolha por camada pode passar de 4
  (o proxy é guloso).
- **GPU.** Só tempo e KL cifrado.

### 2. SparseB no K = 16 com EvalMod de 8 níveis só nessa rota

- **Mecanismo.** O SparseB pousaria 1 nível mais alto (512:24 → 22). As 40 refreshes da softmax ganham 1 unidade cada.
- **Por que só com K = 16.** Medi o K = 24 com grau 27 + 3 ângulos duplos e ele dá **só 10,5-11,8 bits**: inviável
  (`subagent/em_k24_r3.py`). Com K = 16, a tabela em8 da sessão principal dá 23,7 bits.
- **Precedente.** Rodar o SparseB no K = 16 já foi **aceito pelos mantenedores** no Prefill record 3 (BabyStepPrefill:
  `FIDESLIB_BTS_SPARSE_B_K=16`, mesma faixa de inteiros da rota dense, "approximation change" declarada).
- **Onde.**
  - `crates/pc-core/src/bts_setup.rs`: tabela por rota (dense em9, SparseB em8 K = 16).
  - `crates/pc-gpu/src/bts/sparse_b.rs` (`evalmod_k`).
  - Receita: `PLAN_SPARSE_BTS_OUT=1:16,512:22`, com o remap deliberado intocado (o controle `kv22` mostra que o ganho
    vem do SparseB).
- **Medido.** Proxy 96 → **93**. Combinado com o item 1 ficou em 91: não somou no proxy, mas pela garantia acima a ILP
  não piora.
- **Risco.** 23,7 bits de piso absoluto no SparseB. A softmax tolera bem (alvos de 1e-2), mas o piso pesa no fundo da
  banda da rodada 0. Tem que validar na GPU (`pc-bench bts --all`, KL cifrado).
- **Esforço.** 2-4 h em Rust, ILP e bench.

### 3. exp da softmax em intervalo 2× ou 4× com minimax RELATIVO (só config)

- **Mecanismo.** O polinômio do exp passa a cobrir `exp(2^r·a·t)`, e saem r quadraturas. O mapa afim dobrado
  (`SM_FOLD`) não muda.
  - O grau 14 tem a mesma profundidade BSGS do grau 8 do recorde (4).
  - Ajuste IRLS em **erro relativo**: a softmax é invariante a deslocamento por linha, então uma linha com máximo no fundo
    da faixa precisa de precisão relativa lá.
  - O ajuste absoluto (`smwide`) estoura: KL 3,1e-4. Isso confirma que a métrica certa é a relativa (a lente do T1).
- **Medido.**
  - **r=1 em todas: −12 níveis/token.** O erro relativo dentro da faixa fica entre 8e-10 e 7,5e-7: melhor que o recorde
    (2e-6 a 3,7e-4). Fora da faixa (t = −1,157, o caso do recorde na camada 9) também melhora: 4,9e-5 contra 6,6e-4.
    Ou seja, é **ganho puro**.
  - r=2 (−20 a −24 níveis) mantém o KL ≤ 3,9e-6, mas fora da faixa o erro relativo fica grande (camada 9: +5,8 em
    t = −1,157). Exige o range check de 16k prompts que o recorde fez.
  - Proxy: 0 a −1 sozinho. Os níveis caem no segmento push K/V → refresh A, que tem folga. Mas soma nas combinações.
- **Onde.** `configs.json`, chaves `n_squarings`/`delta1`/`cheb_a`/`cheb_b`/`cheb_coeffs` (knob `smrel=` em
  `subagent/search2.py`).
- **Esforço.** < 1 h, mais o range check (CPU).

### 4. GELU em graus que a BSGS do Perseus avalia 1 nível mais rasa (só config)

- **Mecanismo.** O `eval_chebyshev_bsgs` custa ⌈log2 n⌉ + 1 níveis. Para série par + x (a GELU), o traço da
  profundidade dá:
  - grau ≤ 60: 6; grau 62: **7**;
  - grau ≤ 248: 8; grau 254: **9**.
- **Achado.** O comentário em `fit_gpt2_decode16.py` supõe que o grau 63 cabe em 6; na prática o 62 aparado custa 7.
- **Mudança.** Reajustar com o próprio IRLS do recorde: 62 → 60 nas camadas 0, 4, 5, 6, 8, 9 e 254 → 248 nas camadas 2, 11.
  Dá −8 níveis/token.
- **Medido.** KL 4,07e-6. Proxy 95 (−1); com o item 3, 94.
- **Rejeitado.** As camadas 1/3/10 no grau 60 dão KL 1,8e-4.
- **Corolário.** O teste antigo `ln=15` não economizava nível: grau 15 custa 5 + 1 (afim), igual aos graus 18-28. Mas o
  `ln=11` (4 + 1) confirmou 0 de ganho, então a LN segue fora do caminho apertado.
- **Esforço.** 30 min.

### 5. Objetivo da ILP em latência (`PLAN_ILP_OBJECTIVE=ms` + `PLAN_BTS_MS` medido na RTX PRO 6000)

- **Mecanismo.** A receita minimiza a contagem (`total`). O modo `ms` já existe (`placer/ilp.py::_site_cost`) e prefere
  sítios periódicos (SparseB, ~35 % mais barato que dense) com a mesma contagem ou quase.
- **Risco.** Zero de KL e regras.
- **Esforço.** 10 min mais uma ILP. **Não medi:** pede ILP e as latências na GPU.

### 6. Combinação recomendada para a ILP da sessão principal

- **Segura:** `smrel=0` + GELU + adiada {1,3,11} → proxy **92**. Opcionalmente + SparseB K = 16 em8.
- **Agressiva:** r=2 → **91**.
- **Estimativa.** Se a ILP acompanhar o proxy (−4 a −5), os blocos 0-12 vão de 94 para ~89-90. Com o CutMax:
  **~101-102/token contra 106 hoje e 120 no recorde (≈ −15 %)**, mais o que o item 5 render em ms.

## Bom, mas viola as regras ou depende de decisão dos mantenedores

- **CutMax (12/token) — zona cinzenta.**
  - O harness só confere `argmax(z) == argmax(logits)` em ≥ 95 % (`score.py`: `ARGMAX_MIN = 0.95`).
  - Cada iteração do CutMax é monótona (normalização afim positiva + potência ímpar), então menos iterações (ou nenhuma)
    preservam o argmax em aritmética exata. Daria até −12 bts (−10 %).
  - Mas isso esvazia a tarefa "encrypted argmax". Exige uma entrada em `RULINGS.md` antes; eu não submeteria sem ruling.
- **Powerformer (BPMax/BatchLN) e CGF.** Exigem destilação/fine-tuning, o que viola os pesos fixos. O CGF já foi
  descartado por KL.
- **Bootstrap funcional, cortes de grau da GELU, LN mais barata.** Já descartados (≈ 0).

## Resultados negativos (não repetir)

- **EvalMod de 8 níveis no SparseB com K = 24:** 10,5 bits.
- **Remover o refresh deliberado do push K/V:** 101 (+5); o SparseB nos scores não compensa.
- **exp largo com ajuste absoluto:** KL 3e-4.
- **Normalização adiada em todas as camadas:** 95 (piora nas camadas 2, 8, 9, 10).

## Precisa de GPU

- Itens 1 e 2: precisão e KL cifrado; item 2 também com `pc-bench bts --all`.
- Itens 1-5: tempo real (raise drops e mix de rotas mudam o custo por bootstrap).
- Itens 3 e 4 só mudam a config: o KL float64 já está medido acima, falta só o tempo.

## Arquivos (todos novos)

- **Scripts em `subagent/`:**
  - `search2.py`: knobs `smrel=`, `smwide=`, `gelubs`, `gelutr`, `gelu94=`, `--plan-only` e `--no-plan`; o KL e a
    captura estouram 6 GB juntos, então rodo um de cada vez;
  - `plan_count2.sh`: receita com `sed`;
  - `capture_patched.sh` + `patch/{run_capture.py,defer_norm.py,no_kv_bts.py}`;
  - `depth_trace.py` e `depth_even.py`: profundidade da BSGS;
  - `exp_rel.py`, `exp_err.py`, `em_k24_r3.py`, `route_ms.py`;
  - `search2.tsv`.
- **Efeitos colaterais da pipeline:**
  - `work/caps/sa_*`;
  - `work/perseus/bootstrap_placements/cand_sa_*`;
  - `perseus-rec/python/perseus/configs/model/approximation/cand_sa_*`.
- **Lixo a apagar (com aprovação).** Duas pastas criadas pelo bug de word-splitting do zsh (`$spec` sem `${=spec}`, o
  mesmo que gerou as `cand_g127 gelu=127` antigas):
  - `configs/.../cand_sa_s6t2 smsplit=4:0.01:6`;
  - `configs/.../cand_sa_s6t2w2 smsplit=4:0.01:6 smwide=2:14`.

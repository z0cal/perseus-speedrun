# Ganhos com o Trefethen (*Approximation Theory and Approximation Practice*): mapa e prioridades

> **2026-11-02 — base mudou para o recorde 7 (0,968 s/token, cadeia `n32sp`).** Os números abaixo são sobre o recorde 4.
> Sobre o recorde 7: o par z11 + 4 DA (9 níveis) não encaixa na `n32sp` sem redesenhar a cadeia; a variante portada é a
> z7 + 5 DA de mesma profundidade (`PC_EM_EVEN`, 29,53 bits, mesmo pouso), −23,3 ms (−2,4 %) no modelo R7. O SparseB
> K16 já é do upstream (recorde 5). A GELU par dá 212 → 178 produtos/token. Detalhes nas seções de 2026-11-02 do README.

Data: 2026-11-01. Revisão das três notas de `docs/ideias/` (reescritas com estes resultados) contra o que já foi medido
(GPU de 2026-10-30, `tmodel/`, `subagent/ganhos-rapidos/ganhos-rapidos.md`) e contra o código do upstream `d97a6f7` (recorde 4).
**Nada aqui rodou em GPU.** Os ganhos do EvalMod vêm do modelo de tempo calibrado (`tmodel`), aplicado às listas de ops
que o `pc-shadow` do upstream gera para cada série e aos 114 refreshes do plano do recorde 4 (seção 1).

## Resumo

| # | ideia (capítulo do ATAP) | estado | ganho estimado/token | próximo passo |
|---|---|---|---|---|
| 1 | **EvalMod par z11 + 4 DA nas rotas densas, com 1 raise drop** (pouso = recorde) (caps. 3 e 8) | modelo calibrado | **−37,8 ms (−2,2 %)** | implementar + `pc-bench bts` |
| 2 | SparseB par z13 + 4 DA (K=24) com 1 raise drop, no lugar da tabela d27 da opção A (cap. 3) | modelo calibrado | **−22,6 ms (−1,3 %)**; o d27 dá −9,4 | junto com o item 1 |
| 3 | GELU par em z = 2x² − 1 (caps. 3 e 4) | contado (este doc) | −30 produtos ct×ct, ~−4 ms (0,2 %) | só dentro de um pacote |
| 2b | **SparseB com a tabela K16 já existente** (`FIDESLIB_BTS_SPARSE_B_K=16`, achado do teste de sanidade) | modelo calibrado | **−31,2 ms** sem código novo; com o par z11 K16 + drop nas densas e no SparseB, −98,2 ms no total | `bts-prec` + KL |
| 4 | softmax: semente do exp por bandas (caps. 8 e 10); δ2 da camada 4; normalização adiada | **em aberto** | semente ≈ 0 ms; camada 4 ≈ −8,4 ms; normalização adiada 0 a ~−90 ms | ILP + KL (seção 4) |
| — | caps. 23–27 (racionais), f∘arcsin (RBOOT), T1/Newton na LN, CGF | **descartados** | 0 | não repetir |

Os itens 1 + 2 juntos dão **−60,3 ms (−3,6 %)** no modelo, sem mudar a contagem, o posicionamento nem os pousos dos
refreshes. O teste de sanidade independente (`subagent/sanity3/relatorio.md`, Opus xhigh) deu **GO condicional**: confirmou o
pré-passo, as listas de ops, a reconstrução dos sítios e os números do modelo, com as ressalvas da seção 1. Somados à opção C do `tmodel` (1/√ da LN a 1e-3, −19,4 ms, parte do ×1,019 medido), chegam a ~−80 ms
(~−4,7 %), contra a margem de 1 % da regra 5.

## 1. EvalMod par nas rotas densas, gastando o nível livre num raise drop

### O que mudou em relação a `docs/ideias/evalmod-par.md`

A primeira versão de `docs/ideias/evalmod-par.md` (10-30) propunha o par z11 + 4 DA com os planos em9 (108 refreshes). Mas o `tmodel` (seção 2, "efeito de nível")
mostrou por que o em9 perdeu na GPU: **+52 ms** vieram dos pousos 2 limbs mais altos, que encarecem todas as ops a
jusante, e +46 ms do EvalMod d27. Os 6 SparseB a menos devolveram só −31 ms. Com o par z11 o segundo termo vira
ganho, mas o primeiro continua lá. A conta só fecha se o nível livre **não** subir o pouso:

- par z11 + 4 DA = 9 níveis; com +1 raise drop o refresh sai no **mesmo pouso do recorde**;
- o plano fica idêntico ao do recorde 4 (114 refreshes, mesmos posicionamentos e pousos). Muda só o runtime: a tabela
  e o drop. É o mesmo padrão do `tmodel/sbd27.py` (opção A, medida em ×1,007), agora nas rotas densas e StC-first;
- o `tmodel` (seção 3) diz que "toda série de 9 níveis tem mais relins que a enviada (11)". Isso valia para d13 + 5 DA,
  d20 + 4 DA e d27 + 4 DA. **O par z11 + 4 DA é a primeira série de 9 níveis com menos multiplicações que a enviada**
  (9 Mult contra 10 no `em_hist`).

### Conta com o modelo calibrado (`tmodel/even_sites.py`)

1. **Ops de cada série** (`tmodel/emcost_even.rs`, um exemplo do `pc-shadow` do upstream d97a6f7: o censo do
   `emcost_main.rs` mais o pré-passo par). As listas do recorde e do d27 que ele gera são **idênticas** às de
   `tmodel/em/n32a4_t*.jsonl`. As do par estão em `tmodel/em/even_n32a4_t*.jsonl`. No topo 42, cadeia n32a4:

   | série | limbs consumidos | relins | Σlimbs das relins | rescales |
   |---|---|---|---|---|
   | recorde K16 (grau 14 + 5 DA) | 20 | 11 | 359 | 18 |
   | d27 K16 (grau 27 + 4 DA) | 18 | 13 | 439 | 23 |
   | **par z11 + 4 DA (K16)** | **18** | **10** | **334** | **15** |
   | par z13 + 4 DA (K16 ou K24) | 18 | 12 | 400 | 23 |
   | recorde K24 (grau 20 + 5 DA) | 20 | 13 | 423 | 23 |

2. **Sítios do plano do recorde 4** (`bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm`): rota, real ou
   complexo e raise drop de cada um dos 114 refreshes, lidos dos placements (placements explícitos, `hint_fire` e
   deliberados). O mix bate com o do relatório do `tmodel`: 36 StC-first reais, 6 complexos, 40 SparseB-512 e 32 SPRU.
3. **Custo** com `tokplan.site_units` e `calib.json`. O pouso não muda em nenhum cenário, então as ops do plano são as
   mesmas e só os refreshes mudam. **Validação:** o cenário A sai em −9,4 ms, o mesmo valor do relatório do `tmodel`
   para o `sbd27d1_cm`, que foi medido na GPU em ×1,007 (o modelo previa −0,55 %).

| cenário (mesmo plano, mesmos pousos) | Δ refreshes/token | % de 1690 ms |
|---|---|---|
| A: SparseB d27 + 1 drop (`sbd27d1`, medido ×1,007) | −9,4 ms | −0,56 % |
| **A': SparseB par z13 + 1 drop** | **−22,6 ms** | **−1,34 %** |
| densas d27 + 1 drop | +21,0 ms | +1,24 % |
| **E: densas par z11 + 1 drop** | **−37,8 ms** | **−2,24 %** |
| densas par z13 + 1 drop | +6,4 ms | +0,38 % |
| **E + A'** | **−60,3 ms** | **−3,57 %** |
| E + A | −47,2 ms | −2,79 % |

Com +1, os raise drops das rotas densas vão no máximo a 4 (as variantes 1–5 do runtime já cobrem). No SparseB chegam
a 6, a mesma variante que o `sbd27.py` acrescenta.

Limites do modelo: erro de ±6–15 % por etapa, mas as **razões entre séries** são confiáveis (shipped e d27 erram os
dois +8 %). O SPRU fica fixo em 8 ms e não muda nos cenários.

### Ressalvas do teste de sanidade (`subagent/sanity3/relatorio.md`)

- **~25 % do ganho E vem do drop** (−9,4 dos −37,8 ms), com o raise e a CtS 2 limbs mais baixos. Esse efeito só foi
  validado no SparseB (opção A); nas densas é modelo.
- **Desvio conservador no `tokplan`:** o EvalMod do SparseB é precificado 1 nível abaixo do runtime
  (`sparse_b.rs::tops` usa topo 50 − 2d). Com o nível certo, o A' daria ~−27 ms em vez de −22,6.
- **Requisitos que faltavam na lista de passos:**
  - a implementação fica sobre o branch `tmodel-gpu` (34e8db7), que tem `EvalMod` e `shift_evalmod`; o d97a6f7 não
    tem nenhum dos dois;
  - `levels() = 9` no re-pouso do Shadow;
  - `FIDESLIB_BTS_RAISE_DROP=0:1,512:2`. Sem isso, os 12 deliberados K/V por token não acham rota densa com o pouso
    capturado;
  - +1 nos `raise_drop` densos e 512 dos placements, nunca nos SPRU, e a variante 6 em `FIDESLIB_BTS_RAISE_DROPS`;
  - o flag do pré-passo em todos os `build_evalmod`/`program`/`table_bytes` (`dense.rs`, `sparse_b.rs`,
    `sparse_classic.rs`, `pc-tools`).
- **Riscos que o modelo não vê:**
  - a memória das tabelas (o d27 subiu `bts.work` em 1,5 GB no n32);
  - o drop 4 na StC-first, que nunca rodou;
  - a precisão no modo complexo do `bts-prec`, que depende da tabela.
- **Achado do teste: SparseB com K=16.** `FIDESLIB_BTS_SPARSE_B_K=16` já existe no upstream, e o recorde de Prefill 3
  já o usa, declarado como "approximation change". No mesmo modelo:
  - a tabela K16 enviada no SparseB, sem drop e sem código, dá **−31,2 ms**;
  - o par z11 K16 com +1 drop no SparseB dá −60,4 ms;
  - com as densas também em z11 + drop, o total chega a **−98,2 ms (−5,8 %)**, com uma tabela só.

  O risco é a precisão: o K16 no SparseB fica com 27,4 bits, abaixo dos 35,2 do recorde K24. Precisa de `bts-prec` e
  de KL.

### Precisão

O par z11 + 4 DA dá **33,7–34,2 bits** (emulador float64 do teste de sanidade, com o PS real) contra 27,4 do recorde. O raise drop não mexe na
aproximação: só tira um primo que o EvalMod de 9 níveis não usa.

### Negativo desta revisão (não repetir)

Varredura das séries pares de grau baixo com mais ângulos duplos (`em_even.fit_even`, K=16, melhor eps de ajuste):

| série | níveis | bits |
|---|---|---|
| z5 + 5 DA | 9 | 17,7 |
| z7 + 4 DA | 9 | 16,6 |
| z5 + 6 DA | 10 | 27,7 (= recorde, mesmo nível, sem ganho) |
| z7 + 5 DA | 10 | 26,6 |

Em 9 níveis, só z11/z13 + 4 DA chegam à precisão do recorde. Com K=24 (SparseB), nada de grau ≤ 7 passa de 22 bits
em 10 níveis.

### Passos

1. Pré-passo par em `build_evalmod` e variante `EvalMod::Even` em `bts_setup.rs`, como em `docs/ideias/evalmod-par.md`,
   seção "Como implementar". O pré-passo já está escrito em `tmodel/emcost_even.rs`.
2. Raise drop: a receita do recorde 4 já tem `PLAN_RAISE_ROUTES=0,512,1` (dense incluído), e um sítio denso real
   (`v_816`) já usa drop 1. Falta conferir a chave de runtime para dar +1 a todos os sítios densos.
3. Copiar o plano do recorde 4 com +1 drop (como `sbd27.py`) e rodar `compile_check`, `route_mix` e
   `tmodel/predict.py` com os programas compilados (aqui não há CUDA para compilar a `_pc`).
4. GPU: `pc-bench bts` (ms por etapa e `bts-prec`), depois o A/B de `r4_ln3` + item 1 + item 2 contra o recorde 4.

## 2. SparseB: par z13 no lugar do d27 da opção A

A opção A (`sbd27d1_cm`) foi medida em ×1,007 com a tabela d27 K=24 (28,2 bits, 9 níveis) e +1 drop. O par z13 + 4 DA
K=24 tem os mesmos 9 níveis e **30–33 bits**, com 12 relins contra 13 do d27 e do recorde K24. Ele pousa igual, então é
uma troca direta de tabela no mesmo plano: **−22,6 ms no modelo, contra −9,4 do d27** (tabela da seção 1). A
precisão maior também tira o risco "28,2 bits ≪ 35,2 do recorde" que a auditoria apontou.

## 3. GELU par: o ganho real é ~7 vezes menor que o previsto

Contagem com `scripts/gelu_bsgs_count.py`. Ele reimplementa `eval_chebyshev_bsgs` do upstream d97a6f7 sem importar o
Perseus e reproduz a tabela de profundidade de `subagent/ganhos-rapidos/ganhos-rapidos.md` (60 → 6, 62 → 7, 248 → 8, 254 → 9). O
`tmodel/gelu_count.py` fazia a mesma conta só com o tamanho de baby step padrão. A primeira versão de
`docs/ideias/gelu-par.md` não usou nenhum dos dois e previa −170 produtos; a nota foi reescrita com esta tabela.

Config do recorde 4 (`gpt2_base_n32_kc16r_cm`), produtos ct×ct por GELU:

| grau (camadas) | atual (níveis, ct×ct) | par em z, mesmo nível | par em z, 1 nível a menos |
|---|---|---|---|
| 62 (0, 4, 5, 6, 8, 9) | (7, 14) | (7, 12) | (6, 20); em x com m=2 também (6, 20) |
| 60 (7) | (6, 14) | (6, 13) | — |
| 94 (1, 3, 10) | (7, 18) | (7, 15) | — |
| 254 (2, 11) | (9, 29) | (9, 25) | (8, 70) |
| **total/token** | **210** | **180 (−30, −14 %)** | |

- O `IMPL_DEAD_OPS` (ligado por padrão) **já pula** os babies ímpares mortos: os coeficientes ímpares de grau ≥ 3 viram
  exatamente 0 no `_cheb_trim` (ficam em ~1e-16 contra eps 1e-9). Por isso o grau 62 custa 14 produtos e não 26.
- Com 26 ms de produtos e key switches para 210 produtos (`tmodel`), −30 produtos dão **~−4 ms/token (0,2 %)**, não os
  "~170 produtos" e "5 a 10 ms" da nota.
- Achado útil: o grau 62 pode cair para 6 níveis **sem cortar o grau** (com m=2 ou em z: +6 produtos). O `gelubs`
  cortava 62 → 60 e subia o KL (2,7e-6 → 4,1e-6). Só vale se esse nível virar refresh a menos. No recorde 4 o
  `subagent/r4/relatorio.md` diz que o gelubs "não compensa".

## 4. Softmax: o livro ajuda pouco; o que sobra é estrutura

Resumo de `docs/ideias/softmax.md` e da segunda opinião (`subagent/softmax2/relatorio.md`):

- **Semente do exp por bandas (ATAP, caps. 8 e 10).** Nas camadas centradas a métrica certa é mista (absoluta em u ≤ 0),
  porque a referência está sempre na linha.
  - **Alvo:** contra o erro real do recorde, que é dominado pelo fator por linha do Goldschmidt, o (14, 1) passa em
    todas as camadas. O (14, 0), sem quadratura, passa só na camada 4.
  - **Ganho:** ≈ −17 a −19 níveis/token, mas ≈ 0 ms. Os níveis caem no trecho do q, como os do smrel.
  - **Custo real:** coeficientes genéricos; minhas tabelas antigas subcontavam 1–2 produtos. O recorde (8, 2) custa
    (6, 8), o smrel (5, 8) e o (14, 0) (4, 7).
- **Onde pode sair refresh:**
  - **camada 4 com δ2 = 4:** τ = 5,0 já foi recusado pelo upstream; a variante com margem 1,0–1,1 dá −1 SparseB
    (≈ −8,4 ms), com risco de ruído e de NaN;
  - **normalização adiada sobre o recorde 4:** −2 níveis no trecho em que o 3º SparseB de cada bloco pode sumir; teto
    ~−90 ms; nunca foi para a ILP no recorde 4;
  - **Goldschmidt re-dividido** [7, 4] → [6, 5]: mesma profundidade, erro 100–170× menor.

  Tudo isso depende de ILP e KL na máquina do usuário.

## 5. O que o livro NÃO dá (já medido ou descartado)

- **Racionais (caps. 23–27):** cada divisão custa um Goldschmidt (níveis), então perde para polinômio em FHE.
- **f∘arcsin no bootstrap funcional (cap. 8, elipse de Bernstein):** mesmo grau que f para funções suaves, ≈ 0 níveis
  (README, "RBOOT").
- **LN 1/√ com semente + Newton (T1):** nunca ganha nessas faixas (`tmodel`, `ln_t1.py`). O grau cresce com √(b/a) (cap.
  8: ρ ≈ 1 + 2√(a/b)), então só estreitar a faixa reduz grau, e faixa calibrada em dados já foi recusada (risco de NaN).
- **Minimax contra interpolante de Chebyshev (cap. 16):** a diferença é no máximo um fator 2 + (2/π)·log(n+1), ou
  seja, 2 a 3 bits. Os ajustes do recorde já são minimax (IRLS/LP), então não há grau a ganhar aqui.

## Reproduzir

```bash
python3 -I scripts/gelu_bsgs_count.py <perseus>/python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json
```

A varredura de grau baixo da seção 1 usa `em_even.fit_even` / `err_even` com `dz` em 2..7, `r` em 4..6 e eps de ajuste
em {0,5, 2^-4, 2^-6}, reportando o melhor. É uma extensão do laço de `em_even_scan.py` (precisa de numpy e scipy).

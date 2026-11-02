# Vale portar para o recorde 7 quatro ideias medidas sobre o recorde 4? (revisão cética, 2026-11-02)

Base: upstream `speedrun` e805f8d (recorde 7, FlippingToTokenPerSecond, 0,968 s/token, cadeia `n32sp`, 87 refreshes),
clone em `…/scratchpad/up/perseus` (branch `em-gelu-even`, que só mexe no EvalMod e na GELU). Plano servido `_gnabd7`,
configs `gpt2_base_n32_kc16r_cmfl_lhs2_mflp_gn_ab`. Os programas compilados dos 14 estágios vêm de
`…/scratchpad/progs/record7/`, e os grafos de `graphs/gpt2_decode_python_n32sp_kc16r_gnabm`.

Tudo rodou na CPU. Não houve captura, ILP nem GPU: o planner precisa da `_pc` com CUDA. Cada número abaixo vem de uma
de três fontes, e o texto diz qual:
- **medido**: vem dos programas compilados ou do código do upstream (níveis planejados, rotas, drops, contagens);
- **modelo**: vem do modelo R7 (`tmodel/units_r7.py` e `r7_estimate.py`). O absoluto dá 1086 ms contra 968 medidos, e
  o modelo precifica o SparseB-256 como 512, então o valor de um SB-256 é um teto. Dou o valor bruto e o escalado
  (×0,89);
- **suposição**: o que só a captura, a ILP ou a GPU confirmam.

Unidade de nível: a grade da `n32sp` anda de 2 em 2 por nível CKKS. "1 nível" são 2 unidades da grade, e os níveis
dos programas (24…52) estão nessas unidades.

## 0. Resposta curta

| ideia | no recorde 7 | níveis / refreshes | ms/token (modelo R7, bruto) | custo de portar | veredito |
|---|---|---|---|---|---|
| 1. LN adiada | ainda se aplica, mas o código do bundle conflita: a LN do R7 é `_norm_spru`/`_norm_zladder`, e o `norm_defer` traria de volta as escadas | −1 nível por LN (no R4 eram −2); **0 refreshes** | **+3,7** (não monetizado); no R4 a GPU mostrou um resíduo de ~+9,5 ms atribuído a ela | reescrever para spru/zladder, pesos W_C, campos do pc-config, captura, ILP, A/B | **não portar** |
| 2. smrel | o ganho de nível foi anulado: a exp do R7 tem grau 6 (profundidade 3) | **0 níveis líquidos** (+1 em B, −1 em C); 0 refreshes | **−3,3 no melhor caso** (o drop search dá +1 drop ao SB#1 em 9 blocos), **+4,0** se não der | refit ponderado com dados, captura, mapa de decisões, passe de drops | **não portar** (≤ 0,34 %) |
| 3. normalização adiada | quase toda absorvida: `gs_num` já tirou o nível do produto z·r, e o SB#3 que ela mirava não existe mais | −1 nível (no R4 eram −2), só reescrevendo o desempacotamento do `pair_heads`; o fluxo residual o absorve em 9 de 12 blocos; 0 refreshes | **+4 a +10** (+3,8 de produtos extras, mais até +5,8 de ops mais rasas) | reescrever `_softmax_recip` + `complex_softmax_v` + `_unpair_out`, captura, ILP | **não portar** |
| 4. Goldschmidt re-dividido | ainda se aplica igual (camadas 0, 2, 6 e 9 em [7, 4]) | 0 níveis, 0 refreshes | **≈ 0** | só config, mas o fluxo de ops muda, então captura e mapa de decisões | **não portar por tempo**; serve de folga de KL (fator por linha 48–278× menor) |

- **Nenhuma combinação das quatro passa de 1 %:** a melhor soma é −3,3 ms (0,34 %), e só no melhor caso do smrel.
- **Com o EvalMod par:** ele sozinho dá −23,3 ms (−2,4 %). Somado ao smrel no melhor caso, chega a −26,6 ms (−2,7 %).
  A LN adiada e a normalização adiada comeriam a margem.
- **Ordem de prioridade:** 4 (só como folga de KL para outra entrada) > 2 > 1 > 3.

## 1. O que decide tudo: a estrutura de níveis do recorde 7 (medido)

`scripts/sites_r7.py` lista os 87 refreshes com o passo do grafo (log em `sites_r7.log`). Os blocos 0 a 11 têm 6 cada
(os blocos 2 e 11 têm 7):

| refresh | onde | rota | entrada → pouso |
|---|---|---|---|
| SPRU ln_1 | estatísticas da LN (ramo lateral) | SPRU drop 7 | 46–48 → 28 |
| push K/V | `kv_push` (deliberado, pouso fixo) | StC-first drop 0 | 48–50 → 26 |
| **SB#1** | saída da série da exp, **antes** das δ1 quadraturas | SB-256 drop 1–3 | 40 (bloco 4: 50) → 24–28 |
| **SB#2** | **dentro** da rodada 1 do Goldschmidt (o `gs_num` deixa o refresh entrar na rodada) | SB-256 drop 1–2 | **sempre 52** → 24–26 |
| SPRU ln_2 | estatísticas da LN | SPRU drop 7 | 44–52 → 28 |
| denso | saída do `up`, antes da GELU (bloco 2: ln_2; bloco 11: ln_2) | StC-first drop 0–1 | 46–50 → 26–28 |

A softmax de cada camada tem **2** SparseB-256, as camadas 4 e 11 incluídas. No recorde 4 eram 3 por camada, e 5 nas
camadas 4 e 11 (40 SB-512 por token). O "3º SparseB", alvo da normalização adiada no R4, já sumiu.

**Trechos do caminho principal** (`scripts/deficit_r7.py`, `deficit_r7.log`; conteúdo = o que as ops consomem):

| trecho | de → até | conteúdo típico | folga não usada (níveis) |
|---|---|---|---|
| B | pouso do push K/V → SB#1 | 7 níveis (bloco 4: 12) | **5–6** (bloco 4: 1) |
| C | pouso do SB#1 → SB#2 | 12–14 | **0** (SB#2 sempre na entrada 52) |
| D | pouso do SB#2 → denso | 11–13 | 1–4 |
| A | pouso do denso → push K/V seguinte | 11 | 1–2 |

**Fluxo residual** (`scripts/residual_r7.py`, `residual_r7.log`):
- nos blocos 1 e 3 a 10, x_mid = max(out_proj, x_in) = **44 = 44** e x_out = max(down, x_mid) = **44 = 44**;
- a saída do bloco tem teto `level_hint(out, headroom(3))` (`examples/gpt2/block.py:44-51`, `lvl_cap(44)` no grafo).

Uma economia de nível no ramo de atenção antes da soma residual some no máximo, salvo nos blocos 0, 2 e 11.

**Quantos níveis faltam para tirar um refresh** (`scripts/deficit_stream_r7.py`, `deficit_stream_r7.log`):
- o push K/V é deliberado e o denso pré-GELU não pode sair (a GELU e o down são `FORBID_STEPS`);
- o único candidato é um dos dois SparseB, e sobra um SB entre o push K/V (pouso 26) e o x_mid;
- o teto do x_mid é a saída do bloco mais a folga do push seguinte, e a entrada do SB vai até 52 (ou 54, `MAX_LEVEL`).

| | blocos 0, 1 | 2, 3, 9, 10 | 5–8 | 4 |
|---|---|---|---|---|
| faltam (níveis), SB até 52 / 54 | 4 / **3** | 5 / 4 | 6 / 5 | 12 / 11 |

As quatro ideias juntas dão no máximo **2 níveis** nesse caminho por bloco:
- normalização adiada: −1;
- LN adiada no ln_1: +1 nível de teto para a saída do bloco;
- smrel e Goldschmidt: 0.

Não se tira nenhum refresh em nenhum bloco, nem na hipótese otimista (entrada 54, ponto de corte perfeito). Essa
conclusão é medida, mas com uma suposição: que a ILP não ache um corte fora dos pontos que contei. Para tirar um
refresh no R7, falta muito mais nível que no R4, porque os trechos já estão fundidos.

**Sem tirar refresh, a folga só vira tempo como raise drop**, e há três travas:
- **tetos por rota** da guarda `[user_region]` (`crates/pc-program/src/compile/mod.rs:570-596`): denso ≤ 1, já usado
  nos densos pré-GELU dos blocos 2 a 10; SPRU ≤ 7, já em 7 em 23 dos 34; SB ≤ 3 (variantes 1, 2, 3);
- **o drop search** (`python/perseus/plan/placer/planner.py:526-533`) só aceita um drop que deixe "every hint decision,
  every block output and cache-write operand, and every other refresh's landing as they were";
- portanto, folga no trecho D (que chega ao x_mid, a saída do bloco) ou no A (que termina no push K/V deliberado) não
  vira drop. O plano servido já deixa de 1 a 4 níveis em D sem uso. Folga no trecho C (entre SB#1 e SB#2, sem tocar o
  fluxo residual) vira drop no SB#1.

**Preços no modelo R7** (`scripts/r7_levers.py`, `r7_levers.log`, valores brutos):

| item | valor |
|---|---|
| refresh SB-256 | 6,7–7,8 ms (teto) |
| refresh StC-first real | 11,1–11,5 ms |
| refresh SPRU | 6,5 ms |
| +1 drop num SB-256 (refresh mais as ops a jusante 1 nível mais fundas) | −0,6 a −1,3 ms |
| 1 nível economizado e não monetizado (as ops a jusante ficam 1 nível mais rasas, com mais limbs; o máximo nas somas foi propagado) | ln_1 +1,7; ln_2 +2,0; depois da softmax +5,8; trecho C +3,7 ms/token |
| produto ct×ct fundido | 0,05–0,14 ms, conforme o nível |

## 2. LN adiada (`LN_DEFER`, bundle `perseus-lndefer-entry.bundle`, commit 014b81b)

**Aplica-se? Sim, mas o código conflita.**
- O `norm_defer` do bundle é o `norm()` com escadas: a média por `rotsum(xs, t, N)` e a variância por `rotsum(…, 1, N)`.
- No R7, toda LN dos blocos sai antes disso: `norm.py:162-165` devolve `_norm_spru` (10 LNs) ou `_norm_zladder` (14),
  conforme `configs.json` → `norm`.
- Portar o `norm_defer` como está traria de volta as escadas que o recorde 5 tirou (22 + 35 rotações por LN). É preciso
  reescrevê-lo sobre `_norm_spru` / `_norm_zladder`.
- Os pesos W_C (`weights.block_weights(ln_defer)`) e os campos `model.ln_defer` / `ln_defer_exit` (7e497ca) também
  precisam ser portados.

**Quanto economiza no R7: 1 nível por LN, não 2.**
- No R4, o vetor pagava a máscara xs, a média por escada mascarada e o produto: 3 níveis contra 1 adiado.
- No R7, a média vem do ramo já refrescado pelo SPRU, que é raso. Então `centered = xs + mean` (`norm.py:114` e `:145`)
  não custa nível ao vetor: o vetor paga 2 (a máscara e o produto), e a forma adiada paga 1.
- Conferido no traço do bloco 1, ln_2 (`scripts/trace_step.py`): x2 em 42, depois `v_629 = xs + mean(30)` em 42,
  depois `v_631 = centered · inv(40)`, e a saída em 46.

**Onde cai o nível:**
- **ln_1:** no trecho A, que termina no push K/V deliberado (pouso fixo, drop 0 em todos os blocos). A já tem 1–2
  níveis sobrando.
- **ln_2:** no fim de D, que termina no denso pré-GELU, já em drop 1 (o teto) nos blocos 2 a 10. No bloco 2 o denso
  está na própria entrada da LN, então ln_2 nem toca D.
- O único efeito estrutural útil: um ln_1 mais curto permitiria subir o teto da saída do bloco em 1 nível (era o
  `LN_DEFER_EXIT`). Isso entra no cálculo de remoção da seção 1 e não basta: faltam ≥ 3 níveis.

**Ganho:**
- *Medido:* 0 refreshes (a conta de nível acima).
- *Modelo:* +3,7 ms bruto (+3,3 escalado). As ops depois do produto ficam 1 nível mais rasas: o qkv antes do push K/V
  e o matvec do up.
- *Medido no R4 (GPU, fora do harness):* os dois A/B de 2026-10-30 com LN adiada erraram pelo mesmo ~+9,5 ms. O A/B do
  `sbd27d1_cm`, sem LN adiada, bateu com o modelo, então o resíduo foi atribuído à LN adiada
  (`tmodel/RELATORIO-2026-10-30.md` §1 e §6). É a única medida de GPU entre as quatro ideias, e ela é contra.

**Custo de portar:**
- a reescrita da LN (as duas formas), os pesos, os campos de config e o teste de 262 campos;
- captura (CUDA), ILP ou mapa de decisões, passe de drops e A/B.

**Risco:** o KL não preocupa (no R4 a LN adiada deu KL 10× menor, e a conta é exata). O risco é o tempo: o sinal medido
foi positivo.

**Veredito: não portar.**

## 3. smrel (exp com minimax relativo, uma quadratura a menos)

**Aplica-se? O ganho de nível foi anulado pelo que o recorde 7 fez na exp.**
- No R4, a exp tinha grau 8 (profundidade BSGS 4). O smrel trocava por grau 14, na mesma profundidade 4, com uma
  quadratura a menos: −1 nível de graça.
- No R7, todas as 12 camadas usam **grau 6 (profundidade 3)**, com `n_squarings` 2 ou 3, ajuste ponderado por
  p^(2^s−1) e `exp_mask_fold 2` (`configs.json` → `softmax`; `attention.py:887-908`, quadraturas em `:905`).
- `scripts/smrel_r7.py` (log `smrel_r7.log`) mede quanto grau a série precisa para tirar uma quadratura com o mesmo
  erro relativo final: **grau 10–11 em todas as camadas, profundidade 4**. Total da exp: 64 → 64 níveis por token,
  **+0**.
- O resultado é robusto à ponderação: o erro do grau 6 escala com c⁷. Dobrar c multiplica esse erro por ~128, e a
  quadratura a menos só tolera ×2.
- O peso p^(2^s−1) do R7 já é a lente do erro relativo depois das quadraturas: a parte "relativa" do smrel foi
  absorvida.

**Onde cai o nível:**
- a série ganha 1 nível antes do SB#1, no trecho B, que tem 5–6 níveis sobrando: de graça;
- a quadratura que sai fica depois do SB#1, no trecho C, que é o trecho apertado;
- é a única das quatro ideias cujo nível cai entre dois SparseB, onde o drop search consegue usá-lo como +1 drop no SB#1
  sem mexer na saída do bloco;
- os blocos 1, 2 e 9 já têm o SB#1 em drop 3 (o teto).

**Ganho** (*modelo*, `r7_levers.log` §6): por bloco, +2 produtos na série (+0,1 a +0,2 ms), −1 quadratura e +1 drop
no SB#1 onde há espaço.

| cenário | bruto | escalado | % de 968 |
|---|---|---|---|
| melhor caso | −3,3 ms | −3,0 ms | −0,34 % |
| o drop não acontece | +4,0 ms | +3,6 ms | +0,4 % |

*Suposição:* que o drop search pegue esses drops. Ele deveria, porque C fica entre dois refreshes, mas só o planner
confirma.

**Alternativa mais barata, com o mesmo efeito de nível:** pôr o SB#1 depois da primeira quadratura, em vez de antes.
A ILP escolheu "antes", pelo peso de qualidade (`PLAN_QUALITY_WEIGHT=1`). É só planner, sem refit e sem produtos a
mais. As duas precisam da ILP.

**Custo de portar:**
- refit ponderado da exp em grau 10–11 com s−1 quadraturas, sobre as entradas reais do OpenWebText (o ajuste do R7 usa
  os dados dos blocos);
- refazer as folhas do `exp_mask_fold 2`;
- conferir a robustez abaixo de `clip_lo`. O `subagent/softmax2/relatorio.md` §2 achou o problema do
  `CHEB_COEFF_EPS` em grau alto fora de [−1, 1];
- captura, mapa de decisões e passe de drops.

**Risco de KL:** pequeno. No R4 deu KL neutro, e tirar uma quadratura reduz à metade a amplificação do ruído do SB#1.

**Veredito: não portar.** O ganho máximo é ~0,3 %, o sinal depende do planner, e o custo é refit, captura e ILP.

## 4. Normalização adiada (`SoftmaxCfg.defer_norm`, branch `defer-norm` 0b32814 sobre o recorde 3)

**Aplica-se? Quase toda absorvida, e o resto conflita com o `pair_heads`.**
- **No R4, a ideia tirava 2 níveis:** o produto z·(1/s) e a máscara tok0 saíam do caminho, e w = mask(V·z) ia no ramo
  raso, seguido de w·(1/s). O alvo era o SB#3 de cada camada, e o teto estimado era ~−90 ms.
- **O `gs_num` do R7 já tirou o nível do produto.**
  - `attention.py:927-948` chama `goldschmidt_recip(…, num=z)`.
  - Nas palavras do docstring de `poly.py:695-699`: "the result is N/D with no final product (one level and one product
    less)".
  - O branch `defer-norm` mexe exatamente nessas linhas (`y = ops.mult(z, r)` → `return z, r`), então há conflito
    direto.
- **O SB#3 já não existe.** O R7 tem 2 SparseB por softmax (seção 1), com o SB#2 dentro da rodada 1. O teto de −90 ms do
  R4 não vale mais.
- **O que sobra é a máscara.** Com o `pair_heads`, ela é o desempacotamento `_unpair_out` (`attention.py:651-660`,
  chamado em `:710`): uma máscara por deslocamento e depois a rotação.
  - Para adiar o 1/s até depois dele, o 1/s tem de estar no slot de destino.
  - O 1/s está no layout pareado, periódico em tH/2. Nos slots tok0 das cabeças h ≥ H/2, ele carrega a cabeça h − H/2.
  - Isso pede **2 produtos ct×ct (um por deslocamento) e 1 rotação do 1/s por camada**, no lugar do produto que o
    `gs_num` dava de graça.
  - Saldo: **−1 nível** no ramo de atenção.
- **O fluxo residual o absorve** nos blocos 1 e 3 a 10, onde out_proj = x_in = 44 (`residual_r7.log`). O nível só
  chega a jusante nos blocos 0, 2 e 11.

**Ganho:**
- *Medido:* 0 refreshes (seção 1). Mesmo no melhor bloco, o −1 nível não fecha os 3–4 que faltam.
- *Modelo:* produtos e rotação extras **+3,8 ms bruto** (+3,4 escalado); out_proj e o que vier depois 1 nível mais
  rasos, **até +5,8 ms**. Total de **+4 a +10 ms**.

**Custo de portar:**
- reescrever a última rodada de `_softmax_recip`, o ramo `per == 1` com `pair` de `complex_softmax_v`
  (`attention.py:698-710`) e o `_unpair_out`;
- captura e ILP.

**Risco:** no recorde 3, adiar na camada 11 deixou o CutMax inviável.

**Veredito: não portar.**

## 5. Goldschmidt re-dividido (mesma profundidade, menos erro)

**Aplica-se? Sim, igual ao R4.**
- O `smcheb.design` do R7 ainda guarda o primeiro mínimo da grade (`impl/smcheb.py:72`, `sum(ks) < sum(best[0])`
  estrito).
- As camadas 0, 2, 6 e 9 ainda saem com `recip_iters_cheb [7, 4]` e alvo intermediário 0,01 (`configs.json` linhas
  28186, 28380, 28789 e 29065).
- `scripts/gs_r7.py` (log `gs_r7.log`, fórmula exata do `smcheb`, pior kc de 1 a 16):

| camada | servido | fator final \|Σp−1\| | mesma soma, outro alvo | fator | melhora |
|---|---|---|---|---|---|
| 0, 2, 6 | [7, 4] @ 0,01 | 8,2e-4 | [5, 6] @ 0,4 | 2,9e-6 | 278× |
| 9 | [7, 4] @ 0,01 | 8,2e-4 | [6, 5] @ 0,15 | 1,7e-5 | 48× |
| 1, 3, 5, 7, 8, 10 | já na melhor divisão | — | — | — | 1× |

A melhor divisão nas camadas 0, 2 e 6 mudou de [6, 5] (o relatório softmax2 sobre o R4) para [5, 6].

**Ganho:**
- *Medido:* 0 níveis e 0 refreshes. A soma das contagens não muda, e com o `gs_num` o refresh entra em qualquer ponto
  da recursão.
- *Modelo:* ≈ 0 ms. As ops da fronteira entre rodadas (o quadrado, duas `rotate_sum` e a semente) andam 1–2 níveis
  para cima, o que dá algo como +0,1 ms por camada (não calculado em detalhe).
- Não compra nível: para −1 nível (soma 10), o alvo final tem de ir a 1e-2 (erro ×12), linha já descartada no R4.

**Valor real: folga de KL.** Essas 4 camadas têm o maior erro da softmax. O KL do R7 (2,0e-5 de média) está 50× abaixo
do limite, então essa folga só serve para pagar uma mudança que custe KL. O candidato óbvio é a GELU `_ge`, que é
outra função.

**Custo de portar:**
- 4 entradas de config (contagens, alvos; as sementes e as constantes saem do `smcheb`);
- o número de iterações por rodada muda, então muda o fluxo de ops: captura e `map_plan_decisions.py`;
- KL no gêmeo.

**Veredito: não portar por tempo.** Portar junto, de graça em nível, só se outra entrada precisar de KL. Para isso,
**medir antes** o KL do gêmeo com `_ge` + re-divisão contra `_ge` sozinha.

## 6. Prioridade e combinação com o EvalMod par

**Ordem:**
1. Goldschmidt re-dividido: só como companheiro de KL.
2. smrel: o único com sinal de ganho, e no máximo −0,34 %.
3. LN adiada: o sinal medido no R4 é contra.
4. Normalização adiada: absorvida, conflita e piora.

| pacote (modelo R7, bruto) | ms | % de 968 | passa de 1 % (9,7 ms)? |
|---|---|---|---|
| as 4 ideias, melhor caso de cada (smrel −3,3; Goldschmidt 0; LN ≥ +3,7; norm. adiada ≥ +3,8) | só o smrel: −3,3; com as 4: ≈ +4 | — | **não** |
| EvalMod par sozinho | −23,3 | −2,4 % | sim (no modelo) |
| EvalMod par + smrel (melhor caso) | −26,6 | −2,7 % | sim, mas os −3,3 do smrel estão no erro do modelo |
| EvalMod par + LN adiada + norm. adiada | −23,3 + 3,7 + (3,8 a 9,6) = −15,8 a −10,0 | −1,6 a −1,0 % | no limite: encolhe a margem até ~0 |

Os −23,3 ms do EvalMod par também são de modelo (GPU pendente, `gpu/run_gpu_r7.sh`). Nenhuma das quatro ideias melhora
a chance de passar de 1 %.

## 7. Achado lateral (não é uma das quatro ideias): a folga que o plano já tem

O plano servido deixa o SB#2 em drop 1–2 com 1–4 níveis sobrando no trecho D (seção 1). +1 drop no SB#2 dos blocos 0 a
10 vale **até −12,3 ms bruto (−10,9 escalado)** no modelo (`r7_levers.log` §2 e §5): mais que as quatro ideias juntas,
sem mudar nenhuma função.

Ele não acontece porque o x_mid é a saída do bloco: deixar D mais fundo muda a saída, e o drop search
(`planner.py:526-533`) recusa. Para pegar esse ganho é preciso:
- deixar a saída descer 1 nível: `headroom(3)` → `headroom(2)` em `block.py:51`. O push K/V seguinte vai de 50 para 52,
  ainda no teto;
- rodar a ILP e o drop search de novo (precisa da `_pc`);
- medir a precisão do SB-256 em drop 2–3 (`pc-bench bts-prec`).

Tudo isso é hipótese e o valor é um teto (o SB-256 está precificado como 512). Mesmo assim, é uma pergunta melhor para a
próxima rodada do que qualquer uma das quatro ideias.

## 8. O que depende de captura, ILP ou GPU (não roda aqui)

- **Remoção de refresh:** a contagem de níveis da seção 1 vem dos programas. A ILP pode achar um corte que não contei,
  mas faltam ≥ 3 níveis no melhor bloco, e as quatro ideias dão ≤ 2.
- **Drops:** se o drop search pega o +1 drop do SB#1 (smrel) ou do SB#2 (seção 7), só o planner diz.
- **Tempo:** todos os ms são do modelo. O único dado de GPU (a LN adiada no R4) é indireto: é um resíduo de modelo
  atribuído a ela.
- **KL:** nenhuma das quatro foi medida sobre o R7. O Goldschmidt re-dividido e o smrel precisam do gêmeo com os dados.

## 9. Scripts e logs (`subagent/porte-r7/scripts/`)

Rodar com `python3 -I`, com PROGS = `…/scratchpad/progs/record7` e GR = `…/up/perseus/graphs/gpt2_decode_python_n32sp_kc16r_gnabm`.

| script | o que faz | log |
|---|---|---|
| `sites_r7.py PROGS GR GR13` | os 87 refreshes por bloco, com passo, rota, entrada, pouso e drop | `sites_r7.log` |
| `trace_step.py PROGS GR B passos` | traço de passos com os níveis planejados | — |
| `deficit_r7.py PROGS` | conteúdo e folga dos trechos B/C/D/A, e o que falta sem o fluxo residual | `deficit_r7.log` |
| `residual_r7.py PROGS GR` | níveis das duas somas residuais por bloco | `residual_r7.log` |
| `deficit_stream_r7.py PROGS GR` | o que falta para tirar um SB, com o teto do fluxo residual | `deficit_stream_r7.log` |
| `r7_levers.py TMODEL PROGS GR` | modelo R7: custo por rota, valor de +1 drop, nível não monetizado, produtos, folga existente, smrel por bloco, produtos da norm. adiada | `r7_levers.log` |
| `gs_r7.py UP` | divisões do Goldschmidt de mesma soma e fator final, nas configs do R7 | `gs_r7.log` |
| `smrel_r7.py UP` | grau e profundidade para tirar uma quadratura da exp do R7 | `smrel_r7.log` |

# Ideia: EvalMod com série PAR (Trefethen, *Approximation Theory and Approximation Practice*)

Data: 2026-10-30. Status: **só modelo e contagem de operações; nada medido na GPU.**

## Resumo

O alvo do EvalMod antes dos ângulos duplos é par em `u' = u − 1/4`. Basta somar `−1/4` (constante, sem nível),
elevar ao quadrado e usar uma série de Chebyshev em `z` de **metade do grau**. Resultado previsto: o EvalMod de 9
níveis (em9) fica mais barato que o do recorde, em vez de 27 % mais caro como o d27.

## Contexto

- A linha "EvalMod d27 (9 níveis)" foi medida na GPU e fechada: o EvalMod ficou ~27 % mais caro
  (+3,0 ms por refresh dense, +2,2 ms na StC-first), mais do que os 6 refreshes a menos devolvem
  (ver `HANDOFF-2026-10-30.md`).
- O `HANDOFF-2026-10-30.md` pede, na ideia aberta 4, um EvalMod mais barato **no mesmo número de níveis**.
- GELU e softmax não ganham com o livro. A GELU já é um composto ímpar (THOR) ou série ímpar, e o código descarta
  operações mortas. A softmax já usa expoente com quadraturas, minimax relativo e sementes lineares/Chebyshev no
  Goldschmidt. Aproximação racional (caps. 23–27) não serve em FHE, porque a divisão custa níveis.

## A ideia

O alvo do ajuste é `s0 · cos(2π(u − 1/4) / 2^r)` (ver `scripts/em_fit.py`). Ele é par em `u' = u − 1/4`, e a união
de intervalos `{I + δ − 1/4}` é quase simétrica (`|u'| ≤ K − 3/4`). Então:

```
v = (u − 1/4) / K          # AddConst, sem nível
z = 2 v² − 1               # 1 quadrado (+ x2 e −1 no epílogo da relinearização)
p = Σ c_k T_k(z)           # série de grau dz (equivale ao grau 2·dz em u)
… depois os r ângulos duplos, como hoje
```

- Profundidade: `1 (quadrado) + D(dz) + r`, com `D(≤5) = 3`, `D(≤13) = 4`, `D(≤27) = 5`.
- Os fits d27 já indicavam isso: o coeficiente de grau 27 vale 0, então o grau efetivo é 26.
- Um ajuste par em `z` sobre a união de intervalos é equivalente a ajustar o alvo par sobre os intervalos originais.
  Os pontos espelhados não adicionam restrição. A perda de graus de liberdade é pequena.
- Domínio do ajuste: use `L = K`. Com `L = K − 3/4` o ajuste com `eps = 0,5` passa de `|z| = 1` e dá resultados ruins.

## Resultados

Precisão (bits, `-log2` do erro máximo) com `scripts/em_even.py` e `scripts/em_even_scan.py`. Custo com
`crates/pc-shadow/examples/em_hist.rs`. Níveis conferidos com `build_evalmod`: o par z13 + 4 DA dá 18 limbs, igual ao d27.

### K = 16 (dense, StC-first)

| série | níveis | bits | Mult | Rescale | limbs mult / rescale / wsum |
|---|---|---|---|---|---|
| recorde: grau 14 + 5 DA | 10 | 27,4 | 10 | 18 | 428 / 700 / 522 |
| em9 atual: d27 + 4 DA | 9 | 43,5 | 12 | 23 | 514 / 902 / 916 |
| **par z11 + 4 DA** | **9** | **34–35** | **9** | **15** | **400 / 582 / 410** |
| par z13 + 4 DA | 9 | 44,4 | 11 | 23 | 472 / 896 / 304 |
| par z13 + 3 DA | 8 | 23,3 | 10 | 22 | 444 / 866 / 304 |

Validação do proxy: o d27 tem +20 % em Mult, +29 % em Rescale e +75 % em WSum sobre o recorde. A medição na GPU deu
+27 % no EvalMod. Com o mesmo proxy, o **par z11 fica 7 % abaixo em Mult, 17 % em Rescale e 21 % em WSum**, ou seja,
previsão de −10 % a −17 % no EvalMod.

### K = 24 (SparseB, onde ficam os 40 refreshes da softmax)

| série | níveis | bits | Mult | Rescale | limbs mult / rescale / wsum |
|---|---|---|---|---|---|
| recorde: grau 20 + 5 DA | 10 | 35,2 | — | — | — |
| em9 atual: d27 + 4 DA | 9 | 28,2 | 12 | 23 | 514 / 902 / 956 |
| par z13 + 4 DA (ajuste em 2^-6) | 9 | 30–33 | 11 | 23 | 472 / 896 / 304 |
| par z11 + 4 DA | 9 | 23,5 | 9 | 15 | 400 / 582 / 410 |

No SparseB o ganho é menor: o par z13 não é mais barato em Rescale e fica abaixo dos 35,2 bits do recorde.

## Por que a conta pode fechar

- O par z11 pousa igual ao em9 (mesmos 9 níveis, 18 limbs), então os planos `plans/gpt2_decode_em9_cm` e os
  pousos já existentes se aplicam.
- Mesmo número de refreshes do em9 (108 contra 114 do recorde), com EvalMod mais barato que o recorde, em vez de
  27 % mais caro.
- A precisão prevista (34–35 bits) fica acima dos 27,4 bits do recorde na rota dense.

## Riscos e ressalvas

- **Nada foi medido na GPU.** O `time_model` errou no d27. Meça com `pc-bench bts` antes de contar com o ganho.
- O proxy (soma de limbs) não separa o custo de Mult, Rescale e WSum. Só há um ponto de calibração (o d27).
- O quadrado extra adiciona um rescale e um pouco de ruído. Isso precisa de teste de precisão na GPU
  (`bts-prec`) e de KL do modelo inteiro.
- O PS do OpenFHE escolhe `k=2, m=3` para o grau 13, que é um plano caro. O grau 11 usa `k=4, m=2`, que é mais
  barato. Um plano de PS feito à mão para o grau 13 talvez resolvesse.
- As ressalvas já registradas continuam valendo: o teste `oracle_cheb.rs` depende de dump privado, então para
  submissão a tabela precisa ficar atrás de um switch de runtime.

## Como implementar

1. Em `crates/pc-shadow/src/evalmod.rs`, `build_evalmod`: antes de `build_chebyshev`, fazer
   `add_scalar(x, -0.25/K)`, `square(x)`, `add(x, x)` e `add_scalar(x, -1.0)`. O prototype está em `em_hist.rs`
   (3º argumento = K).
2. Em `crates/pc-core/src/bts_setup.rs`: nova variante de `EvalMod` (por exemplo `Even`) com as tabelas
   `fits/even_k16_z11_r4.json` (e a de K=24), `da_its() = 4`, `levels() = 9`, e a flag do pré-passo.
3. Teste `evalmod_series_levels` com a nova variante (esperado: 9 níveis).
4. Switch de runtime (`FIDESLIB_BTS_EVALMOD=even`).
5. Na GPU: `pc-bench bts --all` (ms por etapa e precisão), depois A/B do decode com o plano em9.

## Arquivos

- `scripts/em_even.py`: ajuste LP minimax em `z` e erro.
- `scripts/em_even_scan.py`: varredura de `dz`, `r` e `K`, com bits.
- `fits/even_k16_z11_r4.json`, `fits/even_k16_z13_r4.json`, `fits/even_k24_z13_r4_e6.json`: coeficientes
  (convenção OpenFHE, `c0` não dividido por 2).
- `work/perseus-r4/crates/pc-shadow/examples/em_hist.rs`: histograma de operações e soma de limbs. **Não commitado.**
- `README.md`: seção do fim com a mesma tabela.

## Reproduzir

```bash
cd scripts && python3 -I em_even.py              # tabela de bits
python3 -I em_even_scan.py /tmp/emx              # varredura K=16 e K=24
cd ../work/perseus-r4
cargo run -q --release -p pc-shadow --example em_hist -- ../../fits/even_k16_z11_r4.json 4 16   # par (K=16)
cargo run -q --release -p pc-shadow --example em_hist -- ../../fits/full_d27_r4.json 4          # d27 geral
```

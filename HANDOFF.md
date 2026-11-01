# Passagem — corrigir o plano do Perseus speedrun (Decode-8) antes da GPU

Leia nesta ordem: este arquivo, `README.md` (histórico e números), `subagent/audit/auditoria.md` (auditoria que
reprovou o plano), `work/perseus-rec/README.md` + `work/perseus-rec/CLAUDE.md` (regras do speedrun).

## Estado

- Plano candidato: `plans/fin_dfr13_m_cm` (config `configs/cand_dfr13_cm`) = 86 + 12 = 98 bootstraps/token
  (recorde 120). NÃO VALE ainda: a auditoria achou dois bloqueadores.
- Código: `work/perseus` = branch `evalmod-d27` (EvalMod grau 27 + 4 ângulos duplos, `EVALMOD_LEVELS=9`) +
  commit `defer-norm` (normalização adiada da softmax, `SoftmaxCfg.defer_norm`). `work/perseus-rec` = worktree do
  recorde (a3dff85) com o branch `defer-norm`. Bundle: `perseus-evalmod-d27.bundle`.

## Tarefas

1. **Fazer o plano compilar.** Os pousos foram baixados só na receita do planner; o compilador e o gêmeo de CPU
   ainda têm os do recorde fixos: `crates/pc-program/src/compile/mod.rs:238-240`,
   `crates/pc-shadow/src/route.rs:89-104`, `python/perseus/impl/fakecap.py:144`. O teste Rust
   `route::tests::multi_stage_landings` falha. Colocar o EvalMod novo atrás de um SWITCH de runtime (campo tipado em
   `crates/pc-config`, estilo `FIDESLIB_BTS_*`), com pousos derivados de `EVALMOD_LEVELS` do switch, para que o
   recorde e o Prefill-128 continuem iguais com ele desligado. Critério: `subagent/audit/compile_check.py` dá 0
   erros no plano novo E no do recorde; `cargo test` dos crates de CPU passa (os testes `oracle_*` exigem um dump
   privado dos autores e falham sempre — ignorar).
2. **Recalibrar as faixas da rodada 0 da softmax** (`scripts/sm_split.py`): 16k+ janelas novas do pool OWT
   (inícios de documento + offsets aleatórios), aplicar o topo estrutural do recorde (a docstring diz que aplica, a
   linha 73 não aplica), evitar `l2=0` onde fragiliza (camadas 5,7,8,9,10). NUNCA usar os prompts de dev para
   escolher nada (regra do speedrun; `scripts/headroom_split.py` usava dev — refazer sem). Critério: as 5 janelas
   de estresse de `subagent/audit/band_tail.py` / `kl_windows.py` com KL finito e ~recorde.
3. Refazer o plano: `work/final_plan.sh NOME CONFIG` (posição 0 → ILP base → captura fundida → ILP blocos 0-12 →
   CutMax). NÃO pôr `defer_norm` na camada 11 (o CutMax fica inviável). Rodar compile_check no resultado.
4. Só então: script único para a RTX PRO 6000 (ver "Antes da GPU / Na GPU" na auditoria).

## Armadilhas da máquina (15 GB RAM; o /tmp é tmpfs = RAM)

- Trabalhar só em `~/pesquisa/perseus-speedrun/`. Uma ILP por vez (43 min, 1 thread fixa), cada job pesado sob
  `systemd-run --user --scope -q -p MemoryMax=6G` (a captura fundida com JOBS=2 precisa de 11G). OOM já aconteceu.
- O shell é zsh: `$var` não divide palavras (use `bash -c` em laços com knobs); `scripts/local_env.sh` usa
  `BASH_SOURCE` ⇒ chamar pelo bash; `grep` é função ⇒ `/usr/bin/grep`; `ls` é eza ⇒ `/usr/bin/ls`.
- `env VAR=... função` não funciona (127); `pgrep -f` num laço acha o próprio laço.
- Todo `rm` exige aprovação do usuário (scripts do repo podem limpar os próprios temporários no clone).
- Não acessar servidores sem consentimento. O `ssh sandbox-keml` (hsr) diz no banner "somente para o projeto da
  ANPD" ⇒ não usar sem o usuário confirmar o escopo.
- Commits locais ok (trailer `Co-Authored-By`), nunca push/PR/post: o usuário publica.

## Ferramentas prontas (`scripts/`, `work/`)

`kl_eval.py` (KL float64 com os `ref` do Perseus), `search.py` (knobs: gelu, ln, smtgt, smtop, smsplit, from,
defer), `sm_split.py`, `headroom*.py`, `em_fit.py`/`em_full.py`/`em_k24.py`, `work/capture_rec.sh`,
`work/plan_count.sh` (proxy min-cut 12 s ≈ ILP + 7), `work/final_plan.sh`, `subagent/search2.py`
(knobs smrel, gelubs), `subagent/audit/{compile_check,band_tail,kl_windows}.py`.

## Atualização 2026-10-28

Tarefas 1-3 feitas: ver README, seção "correções da auditoria". O plano candidato agora é `plans/fin_rc3b_m_cm`
(110/token, config `configs/cand_rc3`, commit `d8a7643`). O "98" (`fin_dfr13_m_cm`) é inválido: os 12 pushes K/V
caem na SPRU por remap aplicado duas vezes. Antes de qualquer plano, rodar `scripts/route_mix.py` além do compile_check.

# Handoff — W2 complete, 2026-09-19

Both lanes ran end to end. Nothing is in flight; nothing left a VM or job running.

## Public artifacts

| Artifact | URL |
|---|---|
| Repo (public, branch `main`) | <https://github.com/KodeIsFun/GPUTests> |
| W2 kernel — Kaggle T4 | <https://www.kaggle.com/code/emdadh/free-gpu-bake-off> |
| W2 kernel — P100 requested (got T4) | <https://www.kaggle.com/code/emdadh/bake-off-p100> |
| W2 kernel — slug-mess twin (superseded, kept for provenance) | <https://www.kaggle.com/code/emdadh/free-gpu-bake-off-p100> |
| W2 task (published) | <https://www.kaggle.com/benchmarks/tasks/emdadh/emit-kernel-metadata/1> |

## Live meters (end of W2)

```
GPU       0.05h  29.95h  30.00h  refresh 2026-09-26
TPU       0.00h  20.00h  20.00h  refresh 2026-09-26
Proxy     ~$0.0116 total (W0 $0.00235534 + W2 $0.00928147)
Colab     no active sessions (verified after run)
```

## W2 results

### GPU lane — the bake-off (`projects/2026-W38-bakeoff-t4/results/`)

Same platform-agnostic `job.py` (torch fp16 matmul + whisper-tiny cold load +
30 s synthetic-audio transcribe) on all three lanes:

| Lane | Card delivered | fp16 matmul | whisper RTF | torch |
|---|---|---|---|---|
| Kaggle T4 (`timings-kaggle-t4.json`) | Tesla T4, 15360 MB | 23150.9 | 7.75 | 2.10.0+cu128 |
| Kaggle P100-requested (`timings-kaggle-p100-requested.json`) | **Tesla T4 anyway** | 24358.0 | 9.8 | 2.10.0+cu128 |
| Colab T4 (`timings-colab-t4.json`) | Tesla T4, 15360 MB | **24307.6** | 6.07 | 2.11.0+cu128 (py 3.13) |

1. **Kaggle has no P100 anymore and the CLI does not say so.** A fresh kernel
   whose *first* push carries `--accelerator NvidiaTeslaP100` completes
   silently on a T4. Verified on two kernels (one re-push, one first-push).
2. **Colab free-tier T4 entitlement: VERIFIED** — the open High risk from
   PLAN.md is closed. Real T4, allocation instant at ~21:00 UTC Saturday,
   `colab run` self-cleaned the VM (checked `colab sessions` after).

Known artifact wart: `wall_s` in all three W2 files was measured *before* the
whisper block (dict-literal ordering bug), so it is pre-whisper wall time.
Comparable across lanes, not a whole-job number. Fixed in `job.py` for future
weeks; not re-run (the whisper block carries its own timings).

### Proxy lane — emit-kernel-metadata (`results/eval_results.json`)

5 items × 4 models (same four as W0), deterministic JSON/regex grading: emit a
valid `kernel-metadata.json` for a T4 script kernel, private flip, GPU key
name, 1.5 h / 8 h budgets as `-t` seconds.

| Model | Score | $/item | out tok/item |
|---|---|---|---|
| `gemini-3.5-flash-lite` | **5/5** | $0.00013016 | 46.4 |
| `gemini-3.7-flash` | 5/5 | $0.00142215 | 369.8 |
| `gemini-3.1-flash-lite-preview` | 4/5 | $0.0000778 | 44.0 |
| `gpt-oss-20b` | **3/5** | $0.000226184 | **873.6** |

**17/20, $0.0093 total.** The findings: (a) the schema is common knowledge —
the "literacy" floor is higher than W0's riddle floor suggested; (b) the
*cheapest* model is also the most correct; (c) `gpt-oss-20b` burns 8× the
tokens reasoning and then invents a schema Kaggle never shipped
(`username`+`slug` keys instead of `id`). For W3's `free-gpu-literacy` task:
this item set no longer discriminates at the top — harder items needed (quota
arithmetic, T4-vs-P100 choice, refusal cases).

## Commits

```
(this commit)   W2 bake-off: three free lanes measured, emit-kernel-metadata published
1da4ed4         Reframe the content strategy: friction over spec sheets
e7cb66d..5c6f7be  W0/1
```

75 tests pass. Ledgers: `ledger/runs.jsonl` (6 entries: 4 Kaggle runs + 1
Colab + W0), `ledger/proxy.jsonl` (+4 entries). Committed since 2026-09-19 —
evidence, drafts, and ledgers now travel with the repo (that was open decision
#1, resolved by "commit and push everything").

## Tweet drafts (gate-verified, unposted — X still unwired)

`projects/2026-W38-bakeoff-t4/results/tweets/` — `01-p100-gotcha`,
`02-colab-kaggle-bakeoff`, `03-cheapest-model-wins`, all committed, each
passed `verify_from_files` against its cited evidence files.

## Next: W3 — the 30-hour week

Per PLAN.md Phase 3: public GPU budget template (GPU lane); publish
`free-gpu-literacy` v1 as a public task, then the human clicks "create
Benchmark collection" in the web UI once ≥2 tasks exist (we now have 2:
hello-proxy, emit-kernel-metadata).

```bash
# GPU lane
.venv/bin/python -c "from lab.project import WeekProject, scaffold; ..."
.venv/bin/python -m pytest
# push with the slug guard in force: title must slugify to the metadata id

# Eval lane — always in this order
set -a && source .env && set +a   # kaggle b init -y if LLM_DEFAULT is gone; b auth -y on 401
.venv/bin/python -m lab.task_check
.venv/bin/kaggle b t run <slug> -m <one-model>   # validate solo first
.venv/bin/kaggle b t run <slug> -m <a> -m <b>    # then batch two
.venv/bin/kaggle b t download <slug> -o projects/<wk>/results/
.venv/bin/python -m lab.harvest <dl-dir> <results-dir> <task> <git_sha>
```

## Open decisions for you (one resolved, two live)

1. ~~`.gitignore` excludes `projects/**/results/`~~ — **RESOLVED 2026-09-19:**
   evidence JSONs, raw eval records, tweet drafts, and both ledgers are now
   committed; a reader can check any tweet's digits from GitHub alone.
2. **Benchmark collection** now has 2 published tasks — the one web-UI click
   is due whenever you want it (W3).
3. **X posting still unwired** — drafts accumulate in-repo.

## Gotchas — learned in W2, do not re-learn

Carried from W0/1: colab run --timeout default 30 s; `b t push` runs the task
once against the default model; unschedulable model slugs fail the whole `-m`
list; 1-hour proxy keys; `dont_inherit=True` in task_check; kbench
PassFail-vs-Boolean annotations; tweet-gate "0.64 s" spacing; no `kaggle b
quota` in CLI 2.2.4; week folder = GPU kernel, task slug independent;
`free-gpu-lab` tag invalid; `colab new` does not self-clean.

New:

1. **Kaggle derives the kernel slug from the TITLE, not the metadata id.**
   `title: "Free GPU bake-off (P100)"` + `id: emdadh/bakeoff-p100` creates
   `free-gpu-bake-off-p100`, then every later push of that id 409s. The lab
   now guards this: `write_kernel_metadata` raises `SlugMismatch` unless
   `kaggle_slugify(title) == slug`. Fixed in metadata, not renamed (URLs stay).
2. **`--accelerator` on a re-push does not change the card** — and on W2
   evidence it does not matter even on first push: P100 is gone, silently.
   `build_push_command(accelerator=...)` exists, but verify the delivered card
   in `timings.json` (`gpu` field), never trust the flag.
3. **`kernels status`/`kernels output` need the title-derived slug**
   (`emdadh/free-gpu-bake-off`), not the metadata id slug.
4. **`kernels output` skips download when a local file is newer** — use
   `--force` when re-harvesting a newer version over an old local copy.
5. **Boolean-result tasks record `results[].booleanResult` with null
   `assertions`.** `lab.harvest` now reads both; a scorer that does
   `all([])` on empty assertions will call every item a pass — that bug made
   my first W2 read "20/20" when the truth was 17/20.
6. **`colab` broke between weeks**: uv upgraded `jupyter_kernel_client` to
   1.0.2, which renamed `KernelClient`; colab-cli 0.6.0 dies with
   AttributeError *before* allocating a VM. Fix:
   `uv pip install --python ~/.local/share/uv/tools/google-colab-cli/bin/python 'jupyter_kernel_client<1'`.
7. **`kaggle b auth -y` no longer writes `LLM_DEFAULT`** — after a refresh,
   task_check/import fails with KeyError until `kaggle b init -y` restores it.
8. **The tweet gate cannot verify model names that contain digits**
   ("gemini-3.5" tokenizes as 3.5). Write "Gemini Flash-Lite", or spell the
   number so it traces (46.4, not "about 46").
9. **`job.py` dict-literal ordering**: `wall_s` was computed before
   `details.whisper_block()` ran. Put any "total wall" field last.

## Standing rules still in force

- Max 8 h per Kaggle GPU job; 24 h/week hard stop (6 h of the 30 h kept as buffer).
- Proxy ≤ $3/day default, ≤ $8 on publish days, $90/month hard stop.
- Proxy is **eval-only** — no other caller, ever.
- Every digit in a post must trace to `timings.json` or `eval_results.json`.
- Always `colab stop`; never leave a VM up.

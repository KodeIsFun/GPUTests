# Handoff — W5 complete, month 2 started, 2026-09-20

The month-1 output was calibration, and per the owner's call it is not
marketable content. The pivot is recorded in PLAN.md: month 2 (recipes people
actually want) starts now, at W5. Nothing is in flight; no VM or job is left
running.

## Public artifacts

| Artifact | URL |
|---|---|
| Repo (public, branch `main`) | <https://github.com/KodeIsFun/GPUTests> |
| W5 kernel — GGUF on T4 (3 versions tell the whole story) | <https://www.kaggle.com/code/emdadh/gguf-on-t4> |
| W5 task — recommend-quant-15gb (published, v2) | <https://www.kaggle.com/benchmarks/tasks/emdadh/recommend-quant-15gb/2> |
| W2 kernels + task (calibration, kept) | see git log and PLAN.md |

## Live meters (end of W5)

```
GPU       0.22h  29.78h  30.00h  refresh 2026-09-26
TPU       0.00h  20.00h  20.00h  refresh 2026-09-26
Proxy     $0.0435 lifetime (see ledger/proxy.jsonl for the breakdown)
Colab     not used this week
```

## W5 results

### GPU lane — the GGUF table (`projects/2026-W38-gguf-on-t4/results/timings.json`)

llama.cpp via the **prebuilt cu124 wheel** of llama-cpp-python 0.3.35, 4k ctx,
greedy, all layers offloaded (llama.cpp's own offload lines are in the
committed `kernel-v3.log`):

| Model | File | Load | Decode tok/s | pp512 tok/s |
|---|---|---|---|---|
| Qwen3-4B Q4_K_M | 2.50 GB | 2.0 s | 62.64 | excluded (harness artifact) |
| Qwen3-8B Q4_K_M | 5.03 GB | 2.3 s | 39.55 | 758.08 |
| Qwen3-14B Q4_K_M | 9.00 GB | 3.9 s | 23.21 | 481.89 |
| gpt-oss-20b MXFP4 | 12.11 GB | 57.3 s | **59.21** | 1568.00 |

Headline: **gpt-oss-20b runs on the free T4** — no OOM, 59.21 tok/s, 2.5× the
14B dense. The whole table is the kind of thing people ask every day and
nobody publishes.

### Eval lane — recommend-quant-15gb (`results/eval_results.json`, merged)

Advice graded against our measured table. 14/15 across three models:

| Model | Score | $/item | out tok/item |
|---|---|---|---|
| `gemini-3.5-flash-lite` | 4/5 | $0.0000912 | 28.2 |
| `gemini-3.7-flash` | 5/5 | $0.00207225 | 538.8 |
| `gpt-oss-20b` | 5/5 | $0.0000912684 | 341.8 |

Same 5/5, 22.7× apart in price. flash-lite's single loss is real (the
arithmetic item); gpt-oss-20b's first-run loss was our grader bug (see
gotchas). The eval_results.json is a hand-merge of run1 (flash-lite) + run2
(gpt-oss re-run post grader-fix) — provenance in `merge_note` and in the
archived `recommend-quant-15gb-run1-casebug/` dir.

## Tweet drafts (gate-verified, unposted — X still unwired)

`projects/2026-W38-gguf-on-t4/tweets/` — all pass `verify_from_files` against
`timings.json` + `eval_results.json`:

| File | Shape | Compares |
|---|---|---|
| `01-gpt-oss-runs.txt` | surprise | gpt-oss-20b runs, 59.21 tok/s, vs 14B at 23.21 |
| `02-tok-s-table.txt` | receipt | the whole table |
| `03-same-score-different-bill.txt` | compare | same 5/5, 22.7× price gap |
| `04-wheel-gotcha.txt` | gotcha | source build dies; prebuilt cu124 wheel, 65.8 s |

Gate upgrades this week: `EVIDENCE_KEYS` gained the W5 timing keys
(`tg_tok_s`, `file_gb`, `build_s`, …), and `_collect_evidence` now recurses
into lists of dicts — without that, every per-model number was invisible to
the gate (regression test in `tests/test_tweets.py`).

## Next

- **W6 (image gen)**: SD1.5 vs SDXL-turbo vs Flux-schnell on the same T4 —
  s/image + VRAM; the next "everyone asks" table. Same self-contained-job
  pattern, incremental flush, honest margins.
- **Benchmark collection click** (owner, ~2 min in the web UI): now due —
  three published tasks exist (hello-proxy, emit-kernel-metadata,
  recommend-quant-15gb).
- **X posting is still unwired**; drafts accumulate in-repo. This is the
  plan's whole output goal and remains the standing blocker.

## Gotchas — learned in W5, do not re-learn

Carried from W0–W2: see git history of this file; all still true.

New:

1. **Do not `exec()` a task file locally with fresh proxy auth.** Module-level
   `.evaluate(...)` runs against the live proxy immediately — two accidental
   runs cost ~$0.021 total (logged in `ledger/proxy.jsonl` as
   validation-superseded/final). To unit-test graders, extract functions up to
   `ITEMS = [` and never execute the module.
2. **`kaggle b t download` returns only the current task version's runs.**
   After a re-push, older-version runs stay behind — archive the download dir
   before re-pushing (see `recommend-quant-15gb-run1-casebug/`).
3. **A deterministic grader that fails a right answer is a bug, not
   strictness.** `q4_0` vs `Q4_0` cost a re-push + re-run (~$0.001). Regexes
   grading model output must be case-insensitive and must accept negation
   morphemes ("implausible" is a "no", not a "plausible" miss).
4. **The wheel beats the build on Kaggle:** the source build of
   llama-cpp-python died in 28 s with an unreadable error tail; the abetlen
   cu124 index ships a `py3-none-manylinux` wheel of the same version — 65.8 s
   to working offload. The v3 `nvcc_probe` shows nvcc 12.8 IS present, so the
   build failure cause is genuinely unknown (log was truncated in v1 — full
   logs only from now on).
5. **VRAM pre-check margins must fit the model family.** 3500 MB of margins
   skipped gpt-oss-20b without attempting it (v2); an honest 1500 MB reserve
   (MoE KV at 4k is ~200 MB) let it run (v3). OOM is a result — attempt, catch,
   record.
6. **Prompt-processing benchmarks need a clean context**: run pp first
   (untimed pass pays CUDA graph capture, `reset()`, then timed), or chat
   history pollutes it and you get impossible numbers.
7. **`nvidia-smi memory.used` under-reports llama.cpp VRAM** on the T4 image
   (~55–75% of file size). Use llama.cpp's offload log lines + file size; only
   compare reported MBs relatively.
8. **gpt-oss harmony template does not auto-apply** in llama-cpp-python
   `create_chat_completion`; output carries raw `<|channel|>` markers. Decode
   speed unaffected; content parsing needs the channel format.
9. **The tweet gate saw no numbers inside `"models": [...]`** until the
   collector recursed lists. If a new artifact schema appears, check the gate
   can witness it BEFORE drafting posts.

## Standing rules still in force

- Max 8 h per Kaggle GPU job; 24 h/week hard stop (6 h of the 30 h kept as buffer).
- Proxy ≤ $3/day default, ≤ $8 on publish days, $90/month hard stop.
- Proxy is **eval-only** — no other caller, ever.
- Every digit in a post must trace to `timings.json` or `eval_results.json`.
- Always `colab stop`; never leave a VM up.

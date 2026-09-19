# Handoff — W0/1 complete, 2026-09-19

Both lanes ran end to end. Nothing is in flight; nothing left a VM or job running.

## Public artifacts

| Artifact | URL |
|---|---|
| Repo (public, branch `main`) | <https://github.com/KodeIsFun/GPUTests> |
| W0/1 kernel | <https://www.kaggle.com/code/emdadh/hello-t4> |
| W0/1 task (published) | <https://www.kaggle.com/benchmarks/tasks/emdadh/hello-proxy/1> |

## Live meters

```
GPU       0.01h  29.99h  30.00h  refresh 2026-09-26
TPU       0.00h  20.00h  20.00h  refresh 2026-09-26
Proxy     $0.00235534 total (4 model runs, 20 items)
Colab     no active sessions (verified)
```

Proxy is at **0.0026% of the $90/month cap** ($0.00235534 of $90).

## W0/1 results

### GPU lane — free T4 hardware card

`projects/2026-W38-hello-t4/results/timings.json`

| Field | Measured |
|---|---|
| GPU | Tesla T4 (T4 stuck, not the P100 fallback) |
| VRAM | 15360 MB |
| fp16 matmul | 21162.1 GFLOP/s |
| Driver / CUDA | 580.159.04 / 13.0 |
| torch | 2.10.0+cu128 |
| pip install | 4.39 s |
| git_sha | `eb16cb3` |

### Proxy lane — cost calibration

`projects/2026-W38-hello-t4/results/eval_results.json` — 5 riddles × 4 models,
20 items, all 20 passed, regex-graded (no LLM judge).

| Model | $/item | out tok/item | latency/item |
|---|---|---|---|
| `gemini-3.1-flash-lite-preview` | **$0.0000069** | 1.2 | 0.30 s |
| `gemini-3.5-flash-lite` | $0.0000091 | 1.2 | 0.40 s |
| `gpt-oss-20b` | $0.0000258 | 79 | 0.59 s |
| `gemini-3.7-flash` | **$0.0004293** | 110 | 1.27 s |

**Total $0.00235534.** PLAN.md's priors were wrong by 200–3000×.

Two findings that matter more than the absolute numbers:

1. **Reasoning tokens dominate cost, not prompt length.** `gemini-3.7-flash`
   cost 47× more than `gemini-3.5-flash-lite` on an *identical* prompt, purely
   by emitting 110 output tokens/item instead of 1.2. Model choice moves the
   bill far more than item count.
2. **This is a floor, not a representative number.** Prompts were 20–86 tokens
   with 1-token answers. Real items that emit `kernel-metadata.json` will be
   10–100× larger. Even at 100× it stays in the cents.

## Commits

```
e7cb66d  W0/1 GPU lane complete: Hello T4 measured on a free T4
b002b42  Enforce budget_hours at the platform level; stamp W0 job provenance
eb16cb3  Lab OS: W0/1 scaffold, Colab timeout fix, offline task pre-flight
```

48 tests pass. Ledgers: `ledger/runs.jsonl` (1 entry), `ledger/proxy.jsonl` (4 entries).
Both gitignored by design.

## Next: W2 — the bake-off

Per PLAN.md Phase 3. GPU: same torch/whisper-tiny on Colab T4 vs Kaggle T4 vs
P100. Eval sibling: "emit a valid `kernel-metadata.json` for a T4 script kernel",
graded against the metadata this lab already writes via
`lab.kaggle_runner.write_kernel_metadata`.

```bash
# GPU lane
.venv/bin/python -c "from lab.project import WeekProject, scaffold; ..."
.venv/bin/python -m pytest
.venv/bin/kaggle kernels push -p projects/<wk> --accelerator NvidiaTeslaT4 -t <hours*3600>

# Eval lane — always in this order
.venv/bin/python -m lab.task_check              # free, must pass first
.venv/bin/kaggle b t auth -y                    # 1-hour key; re-run on 401
.venv/bin/kaggle b t run <slug> -m <one-model>  # validate the slug ALONE first
.venv/bin/kaggle b t run <slug> -m <a> -m <b>   # then batch two
.venv/bin/kaggle b t download <slug> -o projects/<wk>/results/
```

**Colab T4 entitlement is still unverified.** W0 never allocated a GPU session —
every Colab call so far used `accelerator=NONE`. This is the one High risk from
PLAN.md that remains open, and W2 is the week that depends on it.

## Open decisions for you

1. **`.gitignore` excludes `projects/**/results/`**, so `timings.json` and
   `eval_results.json` are not in the public repo — a reader cannot check a
   tweet's digits from GitHub alone. Either un-ignore those two files or state
   that Kaggle hosts the evidence. Not changed unilaterally.
2. **Benchmark *collections* need one web-UI click.** `kaggle b leaderboard`
   returns 403 for a bare task; that is expected, not a bug. You have 1 published
   task; PLAN.md W3 wants a collection once there are ≥2.
3. **X posting is still unwired.** `gh`/Kaggle/Colab are all authed, but the
   tweetauto lane was last recorded 402. Drafts are the fallback.

## Gotchas — do not re-learn these

1. **`colab run --timeout` defaults to 30 s.** Kills any real job.
   `lab.colab_runner` always passes it now; raw CLI use still needs it.
2. **`kaggle b t push` is not free** — creating the task runs it once against the
   default model. W0's `gemini-3.7-flash` run came from task creation, before any
   explicit `run`. Budget for the push.
3. **`kaggle b t models` over-reports.** `gpt-5.4-nano` is in the catalog *and*
   the interactive menu but returns *"Failed to schedule runs"*. The error names
   **every** model you passed, not just the bad one — so validate new slugs one at
   a time, never trust the list in the error.
4. **`kaggle b t auth -y` writes a 1-hour key** to `.env`. Required before the
   first `run`; re-run on 401. Already gitignored.
5. **`compile()` inherits `from __future__ import annotations`** from the calling
   module unless `dont_inherit=True`. That silently turned a task's `-> bool` into
   the string `"bool"`, which kbench rejects. Fixed in `lab/task_check.py` — do not
   remove that flag.
6. **kbench writes `<task>-*.run.json` into the CWD** on import. Gitignored, and
   `task_check` runs inside a temp dir.
7. **Default kbench result type is `PassFail`, which expects `-> None`.**
   Annotate `-> bool` for `Boolean`; an unregistered annotation raises at
   decoration time.
8. **The tweet gate needs a space between a number and its unit.** `"0.64s"`
   tokenizes as `0` and is rejected; `"0.64 s"` verifies. Same for `"15360MB"`.
9. **`kaggle b quota` does not exist** in CLI 2.2.4. Track Proxy spend via
   `ledger/proxy.jsonl`.
10. **A week folder is named for its GPU kernel, its task carries its own slug.**
11. **`free-gpu-lab` is not a valid Kaggle kernel tag** — push warns and drops it.
12. **Colab: `colab new` does not self-clean.** An idle session was found burning
    compute units and stopped. Use `colab run` (self-cleaning) with an explicit `-s`.

## Standing rules still in force

- Max 8 h per Kaggle GPU job; 24 h/week hard stop (6 h of the 30 h kept as buffer).
- Proxy ≤ $3/day default, ≤ $8 on publish days, $90/month hard stop.
- Proxy is **eval-only** — no other caller, ever.
- Every digit in a post must trace to `timings.json` or `eval_results.json`.
- Always `colab stop`; never leave a VM up.

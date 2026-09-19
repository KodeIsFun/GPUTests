# Handoff — state as of 2026-09-19 13:53 UTC

Written to pause cleanly. Nothing is running; nothing is mid-flight.

## Where things stand

W0/1 **GPU lane is done and measured**. W0 **Proxy lane is prepared but not run**.
The repo is committed locally but **not yet published to GitHub**.

| Lane | State |
|---|---|
| Kaggle GPU | ✅ Ran, harvested, real numbers below |
| Kaggle Proxy | ⏸ Task file written + validated offline; **$0.00 spent, no run yet** |
| Colab | ✅ Installed, authed, verified, and **idle session cleaned up** |

## Live meters (verified, not estimated)

```
GPU       0.01h  29.99h  30.00h  refresh 2026-09-26
TPU       0.00h  20.00h  20.00h  refresh 2026-09-26
Proxy     $0.00 spent  (no `kaggle b quota` subcommand exists — see gotchas)
Colab     no active sessions (verified clean)
```

The 0.01h is the Hello T4 run: ~36 seconds of an 8h session cap and a 1h
`-t` budget.

## W0/1 result — the hardware card

`projects/2026-W38-hello-t4/results/timings.json`, harvested with no browser:

| Field | Measured |
|---|---|
| GPU | **Tesla T4** (T4 stuck — not the P100 fallback) |
| VRAM | 15360 MB |
| fp16 matmul | **21162.1 GFLOP/s** (4096² ×20) |
| Driver / CUDA | 580.159.04 / 13.0 |
| Compute cap | 7.5 |
| torch | 2.10.0+cu128 |
| pip install (tabulate) | 4.39 s |
| git_sha | `eb16cb3` — provenance stamp works end to end |

Public kernel: <https://www.kaggle.com/code/emdadh/hello-t4>

This closes PLAN.md's "Kaggle API gives P100 not T4" risk for this account, and
gives the first quotable number (21.2 TFLOP/s fp16 on free hardware).

## Commits this session

```
b002b42  Enforce budget_hours at the platform level; stamp W0 job provenance
eb16cb3  Lab OS: W0/1 scaffold, Colab timeout fix, offline task pre-flight
```

`git init` done; `user.email=okemdad@gmail.com`, `user.name=Tuhin`. 48 tests pass.

## Next up, in order

### 1. W0 Proxy calibration — ~$2, the only remaining W0 item

```bash
cd /Users/tuhin/RND/GPUTests
.venv/bin/python -m lab.task_check          # free pre-flight, must pass first
.venv/bin/kaggle b t push hello-proxy -f projects/2026-W38-hello-t4/task.py --wait
.venv/bin/kaggle b t run hello-proxy -m gemini-3.5-flash-lite -m gpt-5.4-nano --wait
.venv/bin/kaggle b t run hello-proxy -m gpt-oss-20b --wait
.venv/bin/kaggle b t download hello-proxy -o projects/2026-W38-hello-t4/results/
```

Two models per `kaggle b t run` (AGENTS.md), so 3 models = 2 invocations.
Confirm the `push` flag spelling with `kaggle b t push --help` first — PLAN.md
records `-f task.py`, which has not been exercised yet.

Then write `ledger/proxy.jsonl` from the actual `Usage` costs. The numbers come
back in **nanodollars** (`input_tokens_cost_nanodollars` /
`output_tokens_cost_nanodollars`, plus `total_backend_latency_ms`) — divide by
1e9 for dollars. This ledger is what every later `estimated_usd` depends on.

### 2. Publish the repo (PLAN.md lock #1 says public)

```bash
gh repo create KodeIsFun/GPUTests --public --source=. --remote=origin --push
```

`gh` is authed as `KodeIsFun` with `repo` scope. Verified: the name is free.

### 3. Corrections to apply to PLAN.md

Its "Current state" table is stale on three rows:

| PLAN.md says | Reality |
|---|---|
| `google-colab-cli` **not installed** | Installed (v0.6.0 via `uv tool`) and authed |
| `gcloud` missing → blocks Colab | Missing, but **irrelevant**: CLI 0.6.0 defaults to `oauth2`, not `adc` |
| Grok *not* promised on the community tier | **Grok is live**: `grok-4.5-0708`, `grok-4.6`, `grok-4.20-reasoning`. OpenAI too (`gpt-5.4`…`gpt-6-astra`). 42 models total |

Also worth folding in: `templates/job.py` still carries the old weak
`gpu_name()` string-matching helper, now superseded by the
`nvidia-smi --query-gpu` approach in the real `job.py`.

## Gotchas found the hard way — do not re-learn these

1. **`colab run --timeout` defaults to 30 s.** Any job longer than 30 s dies.
   `lab.colab_runner` now always passes it (default 5400 s), but raw CLI use
   still needs it.
2. **`kaggle-benchmarks` will not install without `--only-binary=:all:`.**
   A transitive dep (`argon2-cffi-bindings` via `jupyter`) tries to build from
   source and needs CMake ≥ 3.20.
3. **`compile()` inherits `from __future__ import annotations` from the calling
   module** unless you pass `dont_inherit=True`. That silently turned a task's
   `-> bool` annotation into the string `"bool"`, which kbench rejected. This
   bug lives in `lab/task_check.py` and is fixed there — don't remove
   `dont_inherit=True`.
4. **kbench writes `<task>-*.run.json` / `<task>.task.json` into the CWD** when a
   task is imported. Now gitignored, and `task_check` runs inside a temp dir.
5. **The default kbench result type is `PassFail`, which expects `-> None`.**
   Annotating `-> bool` selects `Boolean`. An unregistered annotation raises a
   `TypeError` at decoration time.
6. **`kaggle b quota` does not exist** in CLI 2.2.4 (confirmed — real
   subcommands are `tasks`, `auth`, `init`, `topics`, `leaderboard`). Proxy
   spend must be tracked via the screenshot meters + `ledger/proxy.jsonl`.
7. **A week folder is named for its GPU kernel, its task carries its own slug.**
   `2026-W38-hello-t4/` holds both `job.py` (kernel `hello-t4`) and `task.py`
   (task `hello-proxy`). The lint was refactored to check these independently.
8. **`free-gpu-lab` is not a valid Kaggle kernel tag** — push warns and drops it.
   Cosmetic; fix the `keywords` list in `templates/kernel-metadata.json`.
9. **Colab: `colab new` does not self-clean.** An idle session was found burning
   compute units and was stopped. Use `colab run` (which self-cleans) and always
   pass `-s <name>`.

## Budget ledger state

- `ledger/runs.jsonl` — still empty. The Hello T4 run has **not** been appended;
  add it (0.01h, kernel `emdadh/hello-t4`, git_sha eb16cb3) on resume.
- `ledger/proxy.jsonl` — does not exist yet. First entry comes from step 1.

## Standing rules still in force

- Max 8h per Kaggle GPU job, 24h/week hard stop (6h of the 30h left as buffer).
- Proxy ≤ $3/day default, ≤ $8 on publish days, $90/month hard stop.
- Proxy is **eval-only** — no other caller, ever.
- Every digit in a post must trace to `timings.json` or `eval_results.json`.
- Always `colab stop`; never leave a VM up.

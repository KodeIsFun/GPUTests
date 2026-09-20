# Implementation Plan: Free GPU Lab (Colab + Kaggle, 13 weeks)

## Overview

Stand up a lab in this empty `GPUTests` repo that I can operate without you sitting in a notebook UI. Kaggle is two independent unattended surfaces: **notebook GPUs** (~30 GPU-hours/week) and **Benchmarks Model Proxy** ($10/day, $100/month of LLM inference). Colab is the public "fork this" surface plus a second GPU runner via `google-colab-cli`. Each week is one micro-project (≤7 days) on the GPU lane, plus a small public eval on the Benchmarks lane. Those artifacts become 1–2 X posts/day that build authority because every claim is a run anyone can reproduce on free hardware.

This is not the tweetauto AI-news factory. That pipeline remixes other people's news. This lab ships original GPU work and original model evals. The $100/month is **not** a hidden OpenAI/Anthropic key for YouTube or tweetauto.

## Requirements restatement

- I (Grok) drive Google Colab and Kaggle autonomously after a one-time human auth sitting.
- Timeframe: 13 weeks (~2026-09-19 → 2026-12-18).
- Output: 1–2 X posts every day.
- Authority comes from real work that helps people who also only have free GPUs / free evals.
- Micro-projects only, each ≤7 days, no open-ended training runs.
- Compute is free-tier first: Kaggle GPU + Kaggle Benchmarks + Colab, with Modal/Beam/ZeroGPU as overflow, not as the public face.

## Current state (re-scanned 2026-09-19 after Kaggle install)

Durable copy of this plan: **`/Users/tuhin/RND/GPUTests/PLAN.md`**.

| Item | Status |
|---|---|
| `/Users/tuhin/RND/GPUTests` | `PLAN.md` + Python 3.12 `.venv` with **kaggle 2.2.4** |
| Kaggle CLI | **Working.** Use `.venv/bin/kaggle` (2.2.4). System pyenv still has legacy 1.7.4.5 — do not call that one |
| `~/.kaggle/kaggle.json` | Not needed. 2026 UI "Generate New Token" does not download json |
| `~/.kaggle/access_token` | **Works with CLI 2.2.4.** Authenticated as **emdadh** (Emdadul Hoque). No `kaggle auth login` required |
| GPU / TPU quota (live) | GPU **0.01 / 30h** used (Hello T4 ran), TPU **0 / 20h**, refresh 2026-09-26 |
| Kaggle Benchmarks / Model Proxy | Account has $10/day + $100/month unused. CLI 2.2.4: `kaggle b auth` / `kaggle b init` (there is no `kaggle b quota` subcommand) |
| `gcloud` | **Missing, and not needed.** CLI 0.6.0 defaults to `--auth=oauth2`, not `adc`; the bundled skill says otherwise and is stale |
| `google-colab-cli` | **Installed and authed** — v0.6.0 via `uv tool`, binary at `~/.local/bin/colab`, refresh token in `~/.config/colab-cli/token.json` |
| Python | 3.10.6 via pyenv; lab venv is **3.12.13** |
| Playwright | npx 1.63.0 (fallback only; not the Colab path) |
| GitHub | Logged in as `KodeIsFun` |
| One CLI | 42 connections; **no Twitter/X**; Resend + Google Drive + Browserless exist |
| Composio | Logged in as `okemdad@gmail.com` |
| tweetauto X lane | Built, but last recorded as **402 credits depleted** (2026-09-01) |
| Existing GPU work | `h3-modal` already has measured L4 numbers — usable as Week 8 contrast, not as this lab's runner |

### Kaggle auth gap (2026 UI: json download is the *legacy* button)

You are not missing a hidden download. Kaggle split the API page in 2026:

| Button on https://www.kaggle.com/settings/api | What you get | File? |
|---|---|---|
| **Generate New Token** (API section) | New access token. Copy from the page. Also `KAGGLE_API_TOKEN` / `~/.kaggle/access_token` | **No json download** — this is what you hit |
| **Create Legacy API Key** (under **Legacy API Credentials**) | `{"username","key"}` | **This** is the only button that downloads `kaggle.json` |

We already have `~/.kaggle/access_token` (38 bytes). That is the new-token format. Installed CLI **1.7.4.5 ignores it** and demands `kaggle.json`.

**Recommended (no json file):** upgrade to **kaggle ≥ 2.2** (needs **Python ≥ 3.11** per current CLI docs) in a GPUTests venv, then either:

- use the existing `access_token`, or
- run `kaggle auth login` (browser OAuth, no token to copy), or
- paste the token from Settings into `~/.kaggle/access_token` / `KAGGLE_API_TOKEN`

**If you still want a json:** on that same settings/api page, scroll to **Legacy API Credentials** → **Create Legacy API Key**. That is the download. Then `mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`.

We still need **kaggle ≥ 2.2** either way: 1.7.4.5 has no `kaggle b` (Benchmarks). `kaggle_benchmarks` on PyPI also wants Python ≥ 3.11. Phase 1 therefore starts with a 3.11+ venv in GPUTests, not the system 3.10.6.

## Architecture

Three unattended runners, one control plane:

```
┌─ Mac / Grok session (minutes/day) ──────────────────────────────────────┐
│  Pick week's GPU thesis + eval thesis → push jobs → later pull + tweet  │
└────────┬─────────────────────┬────────────────────────┬─────────────────┘
         ▼                     ▼                        ▼
┌─ Kaggle GPU ─────────┐ ┌─ Kaggle Benchmarks ───┐ ┌─ Colab CLI ─────────┐
│ kernels push          │ │ kaggle b t push/run   │ │ colab run --gpu T4  │
│ --accelerator T4      │ │ Model Proxy LLMs      │ │ dual-publish nb     │
│ 30h GPU + 20h TPU/wk  │ │ $10/day, $100/month   │ │ GPU opportunistic   │
│ `kaggle quota`        │ │ `kaggle b quota`      │ │ always `colab stop` │
└───────────────────────┘ └───────────────────────┘ └─────────────────────┘
         │
         ▼
┌─ Overflow (silent) ─────────────────────────────────────────────────────┐
│ Modal $30/mo · Beam $30/mo · HF ZeroGPU (≤2 Spaces)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this split:** Kaggle's kernel API is the only free GPU I can start, watch, and harvest with a token and no browser. Kaggle Benchmarks is a **separate** product: the `$` quota pays for frontier-model *evaluation* through Kaggle's Model Proxy (`kaggle_benchmarks` SDK, `kaggle b` CLI). It does not add GPU hours and cannot be spent on your apps. Colab has no official "run this notebook while I'm offline" on the free web UI, but `google-colab-cli` is agent-native. Free-tier Colab GPU is still dynamic — a 400 on `colab new --gpu T4` means "no entitlement right now". So Colab GPU is opportunistic; Colab notebooks are always the distribution format.

**Playwright/Selenium against colab.research.google.com is out.** Official CLIs exist.

**Grok cannot be the 3-month daemon.** `scheduler_create` expires after 7 days. GPU jobs and Benchmarks runs live on Kaggle after push. Daily tweets: you open Grok once, or a launchd digest emails drafts via Resend.

## Kaggle AI quota — analysis (the thing you asked to add)

### What it is

The screenshot **$10/day and $100/month "AI quota"** is **not** Kaggle GPU credit, not notebook hours, and not withdrawable money. It is the **Kaggle Benchmarks Model Proxy** allowance: Kaggle pays the upstream inference bill when a `@kbench.task` calls `llm.prompt()` against supported models. Staff and CLI docs treat it as a usage-cost ceiling.

Two different meters:

| Meter | Command | Unit | Recurrence | Pays for |
|---|---|---|---|---|
| Notebook accelerators | `kaggle quota` | GPU / TPU **hours** | Weekly (~30h GPU, 20h TPU) | Your kernel's T4/P100/TPU |
| AI / Model Proxy | `kaggle b quota` | **USD of inference** | Daily $10 + monthly $100 | LLM calls inside Benchmarks tasks |

CLI example (from `kaggle-cli` docs):

```
$ kaggle b quota
period   used    remaining  total    refillAt
Daily    $1.20   $3.80      $5.00    ...
Monthly  $14.50  $85.50     $100.00  ...
```

`remaining = max(0, total - used)`. Hitting $0 means runs fail or queue; it does not charge a card.

### Binding constraint

**The monthly $100 is the real cap, not the daily $10.**

- $10 × 30 days = $300 theoretical, but the month stops you at $100.
- Spend $10 every day → dry on **day 10**, then 20 days of silence (bad for a daily tweet machine).
- Spend $3.20/day average → last the month, with room to burst to $10 on a publish day.

Planning numbers I will actually use:

| Period | Hard cap | Usable plan | Buffer |
|---|---|---|---|
| Calendar day (UTC, per `refillAt`) | $10 | ≤ $8 scheduled | $2 for a surprise rerun |
| Calendar month | $100 | **$90** | $10 |
| 13-week quarter (3 month boundaries) | ~$300 | **$270** | $30 |

If a month starts mid-week-1, I still treat each Kaggle month independently (do not borrow from next month).

### What the proxy can call

Live list only: `kaggle b t models` and `list(kbench.llms.keys())` inside a task notebook. Do not hardcode.

Documented / announced set (as of 2026, mixed sources — verify in W0):

- Community default: **Gemini / Gemma, Anthropic Claude, Qwen, DeepSeek, Z.ai**
- Kaggle (Apr 2026) announced **OpenAI** (e.g. GPT-5.4, GPT-OSS) on Benchmarks
- ~~**Grok** is named on the *Benchmarks Resource Grant*~~ — **wrong as of 2026-09-19: Grok is live on the community tier.** Verified via `kaggle b t models`: `grok-4.5-0708`, `grok-4.6`, `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning`. OpenAI is live too (`gpt-5.4` … `gpt-6-astra`). 42 models in the catalog.
- **Warning: the catalog over-reports.** `kaggle b t models` (and the interactive menu) listed `gpt-5.4-nano`, but scheduling it returns *"Failed to schedule runs"*. Treat the catalog as candidates and confirm schedulability by attempting a run. The error message names **every** model you passed, not just the bad one — so bisect, don't trust the list in the error.
- Docs page still says some models (historically OpenAI) may be missing — the live key list wins

SDK can do: `llm.prompt()`, multi-turn, structured/Pydantic output, images/audio, YouTube-URL video on some Gemini, tool use, sandboxed Python, LLM-as-judge, pandas dataset evals, token/cost/latency metadata (`input_tokens_cost`, `output_tokens_cost`, `total_backend_latency`).

### Cost shape — **measured 2026-09-19** (was: priors)

W0 ran `hello-proxy`, 5 riddles × 4 cheap models = **20 items**, deterministic
regex grading (no LLM judge). Source: `projects/2026-W38-hello-t4/results/eval_results.json`.

| Model | $/item | in tok/item | out tok/item | latency/item |
|---|---|---|---|---|
| `gemini-3.1-flash-lite-preview` | **$0.0000069** | 20 | 1.2 | 0.30 s |
| `gemini-3.5-flash-lite` | $0.0000091 | 20 | 1.2 | 0.40 s |
| `gpt-oss-20b` | $0.0000258 | 86 | 79 | 0.59 s |
| `gemini-3.7-flash` | **$0.0004293** | 20 | 110 | 1.27 s |

**Total for all 20 items: $0.00235534** — 1/850th of the $2 W0 budget.

The priors below were wrong by **200–3000×**. Two findings that matter more than
the absolute numbers:

1. **Reasoning tokens dominate, not prompt length.** `gemini-3.7-flash` cost
   47× more than `gemini-3.5-flash-lite` on an identical prompt, purely from
   emitting 110 output tokens/item instead of 1.2. Model *choice* moves the bill
   far more than item count — so the cheap model must be the default and the
   expensive one a deliberate exception.
2. **These are a floor, not a representative number.** Prompts were 20–86 tokens
   with 1-token answers. Real items that write a `kernel-metadata.json` will be
   10–100× larger. Even scaling 100×, a 50-item single-model eval stays in the
   cents.

Superseded priors, kept only to show the size of the error: short prompt on a
cheap model was estimated at $0.002–$0.03/item (actual: $0.0000069); 50-item ×
1 cheap model was estimated at $0.20–$1.50 (actual: ≈$0.0003). Every
`estimated_usd` from here on is derived from the measured table, not the priors.

### Spend policy (hard rules I will enforce in `lab/quota.py`)

1. Before every `kaggle b t run`, call `kaggle b quota --format json`. Refuse if `daily.remaining < estimated_usd` **or** `monthly.remaining < estimated_usd`.
2. Default daily spend target: **$3**. Publish/leaderboard days may go to **$8**. Never schedule > $8.
3. Multi-model sweeps are **split across days**: 2 models/day, not 6 at once.
4. Develop tasks on the cheapest model; add expensive models only on the freeze day.
5. Judge model ≠ candidate model, and judge is always the cheap one.
6. Dataset evals start at **n=20**. Scale to 50/100 only if the 20-row run came in under estimate.
7. Refresh Model Proxy creds (`kaggle b auth -y`) whenever a run 401s — the key is short-lived.
8. **No proxy call outside a published `@kbench.task`.** Not tweetauto, not faceless-video, not "just one Claude call". If it is not a task with a leaderboard row, it does not run.

### What we will (and will not) use it for

**Will — public Community Benchmarks that help people who also live on free GPUs:**

1. **free-gpu-literacy** — can the model write a valid `kernel-metadata.json`, pick T4 vs P100, and budget hours?
2. **t4-recipe-writer** — given VRAM + a goal (whisper / GGUF / LoRA), does it emit flags that would actually run? Graded against this lab's measured recipes.
3. **gotcha-exam** — items mined from our own OOMs and disconnects (W10 dataset).
4. Optional later: a small **tool-use** task (Python interpreter must compute a VRAM/time estimate, not guess).

**Will not:**

- Hidden API for tweetauto copy, YouTube scripts, agents, or app backends
- Burning Sonnet on 1,000-row evals
- Treating this as $100 of general LLM credit
- Creating Benchmark *collections* via CLI — CLI is **tasks only**; grouping tasks into a Benchmark is a one-time web-UI click (you, ~2 min, when I say the tasks are ready)

### Autonomy path (Benchmarks)

```
kaggle b init -y                         # once per env; writes .env + example
kaggle b auth -y                         # refresh short-lived MODEL_PROXY_*
kaggle b quota --format json             # before every spend
kaggle b t models                        # live model list
kaggle b t push <slug> -f task.py --wait
kaggle b t run  <slug> -m model-a -m model-b --wait
kaggle b t status <slug>
kaggle b t download <slug> -o results/
kaggle b t publish <slug>                # public task + backing notebook
kaggle b leaderboard owner/bench --download
```

Task file rules I will not violate: `# %%` cells, `@kbench.task(name=slug)`, a real `.run(kbench.llm)` or `.evaluate(...)` (missing `.run()` is a silent no-op), return-type annotation if it returns a score, repeat `-d dataset` on every re-push, bare model slugs (`gemini-3.5-flash` not `google/gemini-...`).

Local `python task.py` still burns the **same** quota (it uses the Model Proxy). Use it only for a 1-item validation, not for the full sweep.

**Measured W0 operational facts:**

- **`kaggle b t push` is not free — creating the task runs it once against the
  default model.** W0's `gemini-3.7-flash` run (started 10:24:56, before any
  explicit `run`) came from task creation. Budget for the push itself.
- **`kaggle b t auth -y` writes `.env` with a 1-hour key.** It is not optional
  before the first `run`; `.gitignore` already covers `.env` and
  `**/MODEL_PROXY*`.
- **Two models per `kaggle b t run` works** (`-m a -m b`), matching
  `MAX_MODELS_PER_RUN`. But if *one* name is unschedulable the whole invocation
  is refused, and the error names all of them — so validate new model slugs one
  at a time before batching.
- **The tweet gate needs a space between a number and its unit.**
  `"0.64s"` tokenizes as `0` (the trailing letter breaks the match) and is
  rejected; `"0.64 s"` verifies. Same for `"15360MB"` → write `"15360 MB"`.

## Locks (confirm or change)

1. **Public face:** GitHub repo `KodeIsFun/GPUTests` (or `free-gpu-lab`) + every kernel and every Benchmarks **task** public. — **DONE 2026-09-19:** <https://github.com/KodeIsFun/GPUTests> (public, default branch `main`), kernel <https://www.kaggle.com/code/emdadh/hello-t4>, task <https://www.kaggle.com/benchmarks/tasks/emdadh/hello-proxy>. **Open decision RESOLVED 2026-09-19:** evidence JSONs, raw eval records, tweet drafts, and ledgers are committed — a reader can verify any tweet's digits from GitHub alone.
2. **X account:** tweetauto's EN hero (`ghumaidotcom`) unless you name another aged account. Do not spin a new handle.
3. **Posting:** drafts for 14 days, then auto-post if the verify gate stays clean.
4. **Thesis:** "What actually runs on free Colab/Kaggle this week" **and** "which hosted models can write a free-GPU job that would work" — told as **what broke, what it cost, and what I'd do differently**. Not AI-news remix, not H3 product marketing, and not a spec sheet. Measurements are the evidence, never the headline (see Phase 4).
5. **Human time after Week 0:** ~10 min Sunday + one ~2 min Kaggle UI click when a Benchmark *collection* needs creating. Optional 5 min draft glance during warm-up.
6. **Compute policy:** never pay. GPU hours and Proxy dollars are separate budgets; exhausting one does not unlock the other.
7. **Proxy is eval-only.** No production LLM traffic through Kaggle.

## Provider roles (from the tracker + this quota)

| Provider | Role in this lab | Budget I will actually spend |
|---|---|---|
| **Kaggle GPU** | Primary unattended GPU | ≤18 h experiments + 6 h reruns; 6 h buffer / week. Prefer T4; accept P100 if T4 will not stick |
| **Kaggle Benchmarks** | Primary unattended **LLM eval** | ≤$3 most days, ≤$8 publish days, **$90/month** |
| **Google Colab Free** | Public notebook + opportunistic T4 | Short jobs (≤90 min). Always `colab stop` |
| **Modal $30** | Overflow / 24 GB (H3-class) | Week 8 and GPU-quota emergencies only |
| **Beam $30** | Second overflow | Same |
| **HF ZeroGPU** | Clickable demo in Week 12 (max 2 Spaces) | One Space |
| Saturn / Lightning / Paperspace | Not in the loop | Manual backup |
| Replicate / Baseten / Leonardo / Stability | Ignore | — |
| TPU Research Cloud | Ignore unless a TPU week appears | — |

Kaggle GPU facts: ~30 h/week, 20 TPU-h/week, ~9 h TPU / ~12 h GPU session cap, phone verification, queues. Colab: ≤12 h session, unpublished dynamic GPU, CLI keep-alive 24 h cap (I still stop myself).

## Repo layout (to create in `GPUTests`)

```
GPUTests/
  README.md
  AGENTS.md                      # GPU hours, Proxy $, never leave Colab up, eval-only proxy
  lab/
    kaggle_runner.py             # kernels: init / push / status / output
    benchmarks_runner.py         # kaggle b: quota / auth / push / run / download / publish
    colab_runner.py
    quota.py                     # TWO ledgers: gpu_hours + proxy_usd
    tweets.py                    # from timings.json OR eval_results.json
    project.py
  templates/
    kernel-metadata.json
    job.py
    notebook.ipynb
    task.py                      # @kbench.task skeleton with .run()
    run_manifest.json            # budget_hours AND estimated_usd
  projects/YYYY-Www-slug/
    run_manifest.json
    job.py / notebook.ipynb / kernel-metadata.json
    task.py                      # that week's eval
    results/                     # timings.json + eval_results.json
    tweets/
  ledger/
    runs.jsonl                   # GPU
    proxy.jsonl                  # {utc, usd, model, tokens, task}
  calendar/q4-2026.md
  launchd/
```

Every GPU job writes `results/timings.json`: `{gpu, vram_mb, wall_s, metric, git_sha, platform}`.
Every Benchmarks run writes `results/eval_results.json`: `{task, model, score, passed, input_tokens_cost, output_tokens_cost, latency_s, git_sha}`.
Tweets may only cite digits that appear in one of those two files.

## Pivot 2026-09-20 — content starts at Month 2

Month 1 shipped calibration artifacts: a hardware card, a cost floor, a
three-lane bake-off, a metadata-schema quiz. Every number is real and
reproducible — and none of it is marketable. They are receipts, not content;
per the Phase 4 rewrite below, the marketable thing is the friction people
with free GPUs actually hit. So the calendar skips ahead:

- **W3's** budget template becomes a byproduct of later weeks, not a week of
  its own. Its one live action — publish `free-gpu-literacy` and click the
  Benchmark collection into existence — stays queued until there is a third
  task worth collecting.
- **W4's** whisper week is already half-covered by W2's whisper-tiny RTF
  numbers; large-v3 can slot into a backlog week if it earns its GPU hours.
- **Month 2 starts now, at W5**: "GGUF on T4" — the measured answer to the
  question every free-GPU holder asks first: *which LLM actually runs on this
  thing, and how fast?* Qwen3 4B/8B/14B Q4_K_M + gpt-oss-20b MXFP4, llama.cpp
  on a free T4, download/load/decode/prompt-processed timed per model. Kernel:
  <https://www.kaggle.com/code/emdadh/gguf-on-t4>.

## Implementation phases

### Phase 0 — Human enablement (one sitting, ~35 min)

1. **Kaggle account**
   - Login, phone-verify, enable GPU. Token was created; **Generate New Token does not download json** (2026 UI).
   - Finish auth with **one** of: (a) **Create Legacy API Key** on https://www.kaggle.com/settings/api and drop `kaggle.json` in `~/.kaggle/`, (b) copy the new token into `~/.kaggle/access_token` / `KAGGLE_API_TOKEN`, (c) `kaggle auth login` after we install CLI v2.
   - Confirm the Benchmarks UI shows the $10 / $100 meters (already true per screenshot).
2. **CLIs I will install after you confirm**
   - Python **3.11+ venv** in GPUTests (system is 3.10.6; kaggle 2.2 and `kaggle_benchmarks` want ≥3.11).
   - `pip install -U 'kaggle>=2.2' kaggle-benchmarks` inside that venv (1.7.4.5 has no `kaggle b` and cannot read `access_token`).
   - Smoke: `kaggle kernels list -m --page-size 1` (must not raise `Could not find kaggle.json`)
   - Smoke: `kaggle b init -y` (writes `.env` with `MODEL_PROXY_*`; gitignore it) then `kaggle b quota --format json` and `kaggle b t models`
3. **Colab CLI**
   - Install `gcloud` + `uv tool install google-colab-cli`
   - Browser consent (ADC with `colaboratory` + `userinfo.email` scopes, or `colab --auth=oauth2` paste-code)
   - Smoke: `colab sessions` and a 30s `colab run --gpu T4`. If T4 400s, record it.
4. **X posting** — reconnect Composio Twitter / top up the 402. Confirm handle.
5. **GitHub** — OK a public `KodeIsFun/GPUTests`.

**Risk: High** if skipped.

### Phase 1 — Lab OS (days 1–3 after confirm)

1. Scaffold repo + tests (quota arithmetic for **both** meters, tweet verify gate, task-file lint: decorator + `.run()` present).
2. **Kaggle GPU runner** — `enable_gpu: true`, `enable_internet: true`, public; push with `--accelerator NvidiaTeslaT4`; poll status; pull output; refuse `budget_hours > 8`. Record whether T4 actually sticks vs P100 default.
3. **Kaggle Benchmarks runner**
   - Wrap `quota / auth / push / run / status / download / publish / models`.
   - Estimate USD from W0 calibration before run; abort if over remaining daily or monthly.
   - Split `-m` lists so one invocation cannot exceed $8.
   - Refresh `kaggle b auth -y` on 401.
   - Never pass space-separated `-m a b` (must be `-m a -m b`).
4. **Colab runner** — `colab run` default; always stop; `--config` isolated per project; no interactive `auth`/`drivemount`/`repl`.
5. **Quota ledger** — GPU week-roll at 24 h hard stop; Proxy month-roll at $90 hard stop, day-roll at $8.
6. **Tweet compiler** — two sources (timings / eval_results), same gain-first rules.
7. Optional launchd digest via Resend.

**Risk: Medium.** Dependencies: Phase 0.

### Phase 2 — Daily operating loop

Order of operations when I am in session:

1. `kaggle b quota --format json` + GPU ledger + `colab sessions` + kernel status + `kaggle b t status` for live tasks.
2. Attach `monitor` to anything RUNNING. One GPU job per platform; Benchmarks runs may be 1–2 models at a time.
3. Harvest completed GPU **and** eval artifacts. Update README. Draft 1–2 tweets.
4. Failures: one rerun if queue/disconnect/cheap-model 5xx; otherwise tweet the failure.
5. If under **both** budgets, push the next item in the week's manifest (GPU experiment **or** next 2 models on the eval).
6. Second post: either the other lane's result, or a reply on a large GPU/ML tweet using today's number.

Weekly cadence (dual-track inside the 7-day box):

| Day | GPU lane | Benchmarks lane (≤$3 unless marked) | Tweets |
|---|---|---|---|
| Sun | Thesis + 10-min smoke | 1-item local `python task.py` validation (~$0.05) | Teaser |
| Mon | Experiment A | 2 cheap models, n=20 | GPU cold-start **or** first eval scores |
| Tue | Experiment B | 2 more models | Table |
| Wed | Colab twin | 1 expensive model if remaining ≥ $2 | Notebook drop |
| Thu | Gotcha / ablation | Judge pass on cheap model (only if needed) | Gotcha |
| Fri | Extra seed or rest | **Publish day ≤$8** if the task is frozen | Leaderboard |
| Sat | Freeze GPU results | Idle (protect monthly) | Recap + next week |

That is 7–10 lab posts + replies = 1–2/day. If GPU is queued, the eval lane still produces a post. If Proxy is at $0 for the day, GPU still produces a post.

### Phase 3 — 13-week calendar (GPU project + eval sibling)

Each week still closes in ≤7 days. Eval tasks are small on purpose so they cannot steal the week.

#### Month 1 — Map the free GPUs + spend $90 of Proxy on a literacy bench

| Week | GPU project | Benchmarks sibling (Proxy) | Success |
|---|---|---|---|
| **W0/1 Hello T4** | nvidia-smi, CUDA, pip times, Colab vs Kaggle | **Hello Proxy:** 5 riddles × 3 cheap models, **budget $2**, write `proxy_costs.jsonl` | Hardware card + `$` per call card |
| **W2 Bake-off** | Same torch/whisper-tiny on Colab T4 vs Kaggle T4 vs P100 | Task: "emit valid kernel-metadata.json for a T4 script kernel" | Wall-time table + pass/fail by model |
| **W3 30-hour week** | Public GPU budget template | **Publish v1 public task** `free-gpu-literacy` (metadata + quota arithmetic items). You click "create Benchmark" in the UI | Gist + public task URL |
| **W4 faster-whisper** | large-v3 RTF on T4 | Task: pick batch/compute-type given 15 GB (graded against W4 measurements) | RTF table + model accuracy vs our recipe |

Month 1 Proxy plan: ~$20 calibration + ~$40 literacy bench + ~$20 whisper-flags + $10 buffer ≈ **$90**.

#### Month 2 — Recipes, evals graded against *our* numbers

| Week | GPU project | Benchmarks sibling | Success |
|---|---|---|---|
| **W5 GGUF on T4** | 7B/8B/14B Q4/Q5 tok/s | "Recommend quant given 15 GB" — answers checked against W5 table | tok/s table; 14B runs or we publish the OOM |
| **W6 Image gen** | SD1.5 vs SDXL-turbo vs Flux-schnell | VRAM-estimator questions (n=20) | s/image + VRAM; Flux yes/no |
| **W7 QLoRA 7B** | One LoRA in ≤9 h | "Will this finish before the 9 h kill?" reasoning + numeric guess | Adapter **or** died-at epoch/hour |
| **W8 Video, honestly** | What fits T4 vs `h3-modal` L4 numbers | Skip expensive eval this week (protect $). Tiny license/territory *reading* task only if remaining ≥ $15 | T4 can/cannot table; no H3 product publish if territory still gray |

Month 2 Proxy: ~$70 on recipe-writer items + $20 buffer.

#### Month 3 — Compound

| Week | GPU project | Benchmarks sibling | Success |
|---|---|---|---|
| **W9 Embeddings** | e5/gte throughput | Refresh `t4-recipe-writer` with W5–W9 items | docs/s table |
| **W10 Gotchas dataset** | Publish OOMs/disconnects as a Kaggle dataset | **gotcha-exam** graded on that dataset (attach with `-d` every push) | Public dataset + public task |
| **W11 One-click kernel** | Best recipe as Run-All notebook | "Improve this kernel" — model output must still pass a smoke assertion | 3-cell happy path |
| **W12 ZeroGPU Space** | Wrap W11 in one Space | Optional multimodal (image of `nvidia-smi`) if a vision model is in `t models` and remaining ≥ $10 | Space URL |
| **W13 Recap** | Open `runs.jsonl` + `proxy.jsonl` | Leaderboard CSV dump of all our tasks; spend report: $ used vs $100 × 3 | Recap thread; repo is the product |

Backlog substitutes: TPU v5e vs GPU; ONNX vs llama.cpp; community-requested eval week; Benchmarks Resource Grant application **only if** a task is getting real forks (do not apply in W1).

**Risk: Medium** (downloads eating GPU hours; Proxy under-estimated). Mitigation: weights as Kaggle datasets; n=20 first; abort over estimate.

### Phase 4 — X authority system

**Rewritten 2026-09-19.** The first drafts were spec-sheet posts ("15360 MB VRAM,
21162.1 GFLOP/s fp16"). They were true, verified, and useless: no reader has a
reason to care, and it reads as a nerd reciting numbers. A measurement is not a
story. The interesting thing is never the number — it is the **friction, the
surprise, or the fix**.

**Lead with the thing that costs the reader something if they don't know it.**

Three post shapes, in priority order:

1. **The gotcha** — an undocumented behaviour that wastes an afternoon.
   *"Kaggle's model list is a catalogue, not a promise: it lists a model as
   available, then refuses to schedule it — and the error blames every model you
   passed, not the broken one."* This is the highest-value content and it needs
   no number at all.
2. **The surprise** — what we expected vs what we measured.
   *"Budgeted a couple of dollars to find out what evals cost. Cheapest
   $0.0000069 a question, dearest $0.0004293. I was off by three orders of
   magnitude."* The number is evidence for the surprise, never the headline.
3. **The receipt** — the public kernel or task, offered rarely, so readers can
   check the work. Credibility, not content.

**Voice rules:**

- No units the reader has to decode. GFLOP/s, token counts, and nanodollars are
  evidence, not copy.
- Every post must survive "so what?" and "says who?". The "so what" is the
  lesson; the "says who" is the public artifact.
- Prefer plain constructions. "Two models, same question, one cost 62× more" beats
  "we observe significant cost heterogeneity across model families".
- Own the mistakes. Being wrong by 3000× and saying so is more credible than
  never being wrong.

**The gate stays.** Two legal number sources: `timings.json` (GPU) and
`eval_results.json` / `eval_results.<model>.json` (Proxy), merged via
`verify_from_files(tweet, *paths)`. Every digit must still come from a real
artifact — that constraint is what makes the "check my work" claim honest, and it
does not limit the voice above, which is deliberately number-light.

- Warm-up 1/day week 1, 1–2/day from week 2.
- Replies: 1/day on large GPU/ML threads, answering with this week's gotcha or
  measured surprise — not with a spec sheet.
- Pin: W2 bake-off, then W3 public task, then W11/W13.
- Bio: "Weekly notes from a free-GPU lab. What broke, what it cost, what I'd do
  differently. Everything reproducible."

If X is still 402, drafts continue. Lab does not stall.

## Testing strategy

- Unit: GPU week-roll, Proxy day/month roll, tweet verify against **both** JSON files, task-file lint (decorator + `.run()`).
- Integration W0: live Kaggle GPU hello-world, live `kaggle b t run` $2 smoke, live Colab CLI hello-world.
- E2E per week: GPU push→harvest **and/or** `b t run`→download; README row; tweet draft.
- Negative: refuse 10 h GPU job; refuse run when `daily.remaining < estimate`; refuse tweet whose `$1.40` is not in `eval_results.json`; `colab stop` on failure.

## Risks and mitigations

| Risk | Level | Mitigation |
|---|---|---|
| Burning $100 by day 10 | High | $3/day default, $90/month hard stop in code |
| Colab free tier refuses T4 | High | Colab = public notebook; numbers from Kaggle GPU |
| Kaggle API gives P100 not T4 | ~~Medium~~ **Closed** | W0 got a real **Tesla T4** (15360 MB, 21162 GFLOP/s fp16) with `--accelerator NvidiaTeslaT4` |
| GPU phone-verify / queue | High | Phase 0; overnight; eval lane still tweets |
| Downloads eat 30 h | High | Dataset-pin weights |
| Proxy key expires mid-run | Medium | `kaggle b auth -y` on 401 |
| Missing `.run()` silent no-op | High | Lint in `benchmarks_runner.py` before push |
| Using Proxy as a secret API | High | AGENTS.md + code path only through `@kbench.task`; no other callers |
| OpenAI/Grok not on community list | Low | Live `t models`; never advertise a model that is not in the list |
| LLM-as-judge doubles cost | Medium | Cheap judge only; skip judge if assertions suffice |
| Benchmark *collection* needs UI | Low | You click once when I have ≥2 published tasks |
| H3 territory clause | High for W8 | T4 feasibility only; no product |
| Grok scheduler 7-day expiry | High | Work lives on Kaggle |
| X 402 | High | Drafts + Resend digest |
| Idle Colab VM | High | `colab run`; always stop |

## Success criteria (end of week 13)

- [ ] I can start, watch, and harvest a Kaggle GPU job with no browser.
- [ ] I can push, run, download, and publish a Kaggle Benchmarks task with no browser (collection grouping may need one UI click).
- [ ] I can start, run, download, and stop a Colab CLI job with no browser.
- [ ] 13 public kernels + matching Colab notebooks.
- [ ] ≥3 public Benchmarks tasks and ≥1 Benchmark collection, with downloadable leaderboards.
- [ ] ≥90 original posts, each digit-traceable to `timings.json` or `eval_results.json`.
- [ ] Community artifacts: W3 budget template, W4 whisper kernel, W3/W10 public evals, W10 gotchas dataset, W11 one-click kernel, W12 Space.
- [ ] GPU spend ≤ 30 h/week; Proxy spend ≤ $90/month; no paid invoices.
- [ ] **Zero** Proxy calls into tweetauto / video pipelines.
- [ ] Human time ≤10 min/week after Phase 0 (plus rare UI click).

## What I will not do

- Multi-week training, or "build a product on Colab".
- Browser-automate the Colab website.
- Invent GPU numbers or eval scores.
- Spend Modal/Beam on content a T4 can produce.
- Spend Kaggle Benchmarks quota on anything that is not a public `@kbench.task`.
- Duplicate tweetauto's AI-news voice on this account.

## Estimated complexity

- Phase 0 (you): ~35 min, blocked on auth.
- Phase 1 (me): 1–2 days.
- Phase 2–4: 13 weeks, ~1–2 hours of my session time per week, unattended GPU hours, and ~$90/month of already-granted Proxy.

**WAITING FOR CONFIRMATION:** Proceed with this plan? Reply `yes`, `yes, with: …`, or `modify: …`. I will not create the repo, install CLIs, or spend GPU hours or Proxy dollars until you do.

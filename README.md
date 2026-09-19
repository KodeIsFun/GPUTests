# Free GPU Lab

Unattended measurements on **Kaggle GPU** (~30h/week) and **Kaggle Benchmarks** (Model Proxy), with Colab as the public notebook surface. Every number below is measured, not estimated.

**W0/1 is complete** — both lanes ran end to end with no browser.

| Meter | Live (2026-09-19) |
|---|---|
| Kaggle GPU | **0.01 / 30h** (refresh 2026-09-26) |
| Kaggle TPU | 0 / 20h |
| Benchmarks Proxy | **$0.00235534** spent of a $90/month hard stop |
| Colab | no active sessions |

## W0/1 results

### Free Kaggle T4 — hardware card

Kernel: [emdadh/hello-t4](https://www.kaggle.com/code/emdadh/hello-t4)

| Field | Measured |
|---|---|
| GPU | Tesla T4 (T4 stuck, not the P100 fallback) |
| VRAM | 15360 MB |
| fp16 matmul | 21162.1 GFLOP/s |
| CUDA / driver | 13.0 / 580.159.04 |
| pip install | 4.39 s |

### Model Proxy — cost calibration

Task: [emdadh/hello-proxy](https://www.kaggle.com/benchmarks/tasks/emdadh/hello-proxy/1) — 5 riddles × 4 models, 20 items, regex-graded (no LLM judge).

| Model | $/item | out tok/item | latency/item |
|---|---|---|---|
| `gemini-3.1-flash-lite-preview` | **$0.0000069** | 1.2 | 0.30 s |
| `gemini-3.5-flash-lite` | $0.0000091 | 1.2 | 0.40 s |
| `gpt-oss-20b` | $0.0000258 | 79 | 0.59 s |
| `gemini-3.7-flash` | **$0.0004293** | 110 | 1.27 s |

All 20 items passed. Total: **$0.00235534**.

**Reasoning tokens dominate cost, not prompt length** — `gemini-3.7-flash` cost 47× more than `gemini-3.5-flash-lite` on an identical prompt, purely by emitting 110 output tokens per item instead of 1.2. These are short-prompt numbers and therefore a *floor*: real items that emit `kernel-metadata.json` will be 10–100× larger.

Evidence: `projects/2026-W38-hello-t4/results/{timings.json,eval_results.json}` (gitignored — see open decisions in [HANDOFF.md](HANDOFF.md)).

## Documentation

- [PLAN.md](PLAN.md) — the 13-week plan
- [AGENTS.md](AGENTS.md) — hard budget stops and agent rules
- [HANDOFF.md](HANDOFF.md) — current state, next steps, operational gotchas

## Working in this repo

```bash
source .venv/bin/activate                # Python 3.12 + kaggle 2.2.4
pytest                                   # 48 tests

.venv/bin/kaggle quota                   # GPU/TPU hours
.venv/bin/python -m lab.task_check       # free pre-flight for a Benchmarks task file

# before any GPU push — never skip
.venv/bin/python -c "from lab.quota import *; ..."   # decide_gpu_job(quota, budget_hours)
```

Kaggle user `emdadh`; CLI is `.venv/bin/kaggle`, never the system 1.7.4.5 shim.
Colab CLI is `~/.local/bin/colab` (google-colab-cli 0.6.0, `--auth=oauth2`).

## Budget rules (enforced in code, not by habit)

| Resource | Limit | Where it is enforced |
|---|---|---|
| Kaggle GPU | 8 h/job, 24 h/week hard stop | `lab/quota.py:decide_gpu_job`, plus `kernels push -t` at the platform |
| Proxy daily | $3 default, $8 publish day | `lab/quota.py:decide_proxy_run` |
| Proxy monthly | $90 hard stop | `lab/quota.py:decide_proxy_run` |
| Proxy scope | eval-only, never a general API | `tests/test_project_tasks.py` rejects direct provider SDKs |
| Colab | never leave a VM up | `lab/colab_runner.py` forbids interactive subcommands |

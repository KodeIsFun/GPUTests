# Free GPU Lab

Unattended measurements on **Kaggle GPU** (~30h/week) and **Kaggle Benchmarks** (Model Proxy), with Colab as the public notebook surface. Every number below is measured, not estimated.

**W0/1 and W2 are complete** — both lanes ran end to end with no browser.

| Meter | Live (2026-09-19) |
|---|---|
| Kaggle GPU | **0.05 / 30h** (refresh 2026-09-26) |
| Kaggle TPU | 0 / 20h |
| Benchmarks Proxy | **$0.0116** spent of a $90/month hard stop |
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

## W2 results — the bake-off

### GPU lanes: same torch + whisper-tiny, three free surfaces

Kernels: [emdadh/free-gpu-bake-off](https://www.kaggle.com/code/emdadh/free-gpu-bake-off) (Kaggle T4), [emdadh/bake-off-p100](https://www.kaggle.com/code/emdadh/bake-off-p100) (P100 requested), Colab `colab run --gpu T4`.

| Lane | Card delivered | fp16 matmul | whisper-tiny RTF | torch |
|---|---|---|---|---|
| Kaggle, T4 requested | Tesla T4 (15360 MB) | 23150.9 GFLOP/s | 7.75 | 2.10.0+cu128 |
| Kaggle, **P100 requested** | **Tesla T4 anyway** | 24358.0 GFLOP/s | 9.8 | 2.10.0+cu128 |
| Colab, T4 requested | Tesla T4 (15360 MB) | **24307.6 GFLOP/s** | 6.07 | 2.11.0+cu128 |

Two findings:

1. **There is no P100 anymore.** A brand-new kernel whose *first* push carries `--accelerator NvidiaTeslaP100` completes silently on a Tesla T4. No warning, no error — the CLI accepts the flag and Kaggle hands you different silicon. Any 2023 recipe tuned on P100 quirks is running on a T4 now.
2. **Colab's free T4 entitlement is real** (verified 2026-09-19, was W0's one open High risk). Same card name as Kaggle, slightly *faster* on the same matmul, newer torch — and the VM self-cleans after `colab run`.

### Eval sibling: emit-kernel-metadata

Task: [emdadh/emit-kernel-metadata](https://www.kaggle.com/benchmarks/tasks/emdadh/emit-kernel-metadata/1) — 5 items × 4 models (same four as W0), deterministic JSON/regex grading: emit a valid `kernel-metadata.json` for a T4 script kernel, flip `is_private`, name the GPU key, convert 1.5 h and 8 h budgets to `-t` seconds.

| Model | Score | $/item | out tok/item |
|---|---|---|---|
| `gemini-3.5-flash-lite` | **5/5** | $0.00013016 | 46.4 |
| `gemini-3.7-flash` | 5/5 | $0.00142215 | 369.8 |
| `gemini-3.1-flash-lite-preview` | 4/5 | $0.0000778 | 44.0 |
| `gpt-oss-20b` | **3/5** | $0.000226184 | **873.6** |

17/20 overall, **$0.0093 total**. The cheap model that wins: `gemini-3.5-flash-lite` — perfect schema, 46 output tokens/item. The loser is the open-weights model: `gpt-oss-20b` emitted 873.6 tokens/item of reasoning and still "remembers" a schema Kaggle never shipped (`username`+`slug` keys instead of `id`), failing both metadata items.

Evidence: `projects/2026-W38-bakeoff-t4/results/` (gitignored — see open decisions in [HANDOFF.md](HANDOFF.md)).

## Documentation

- [PLAN.md](PLAN.md) — the 13-week plan
- [AGENTS.md](AGENTS.md) — hard budget stops and agent rules
- [HANDOFF.md](HANDOFF.md) — current state, next steps, operational gotchas

## Working in this repo

```bash
source .venv/bin/activate                # Python 3.12 + kaggle 2.2.4
pytest                                   # 75 tests

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

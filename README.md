# Free GPU Lab

Unattended measurements on **Kaggle GPU** (~30h/week) and **Kaggle Benchmarks** (Model Proxy), with Colab as the public notebook surface. Every number below is measured, not estimated.

**Month 1 was calibration** (hardware cards, cost floors, a bake-off). **The content starts here, at Month 2:** the questions people with free GPUs actually ask, answered with runs they can reproduce.

| Meter | Live (2026-09-20) |
|---|---|
| Kaggle GPU | **0.22 / 30h** (refresh 2026-09-26) |
| Kaggle TPU | 0 / 20h |
| Benchmarks Proxy | **$0.043** spent this month of a $90 hard stop |
| Colab | no active sessions |

## W5 results — which LLM actually runs on the free T4, and how fast

The first table everyone looks for and nobody publishes. Kernel: [emdadh/gguf-on-t4](https://www.kaggle.com/code/emdadh/gguf-on-t4) — llama.cpp (llama-cpp-python 0.3.35, prebuilt cu124 wheel), 4k context, all layers on GPU, greedy decode.

| Model | File | Download | Load | Decode (tok/s) | Prompt 512 (tok/s) |
|---|---|---|---|---|---|
| Qwen3-4B Q4_K_M | 2.50 GB | 13 s | 2.0 s | **62.64** | n/a (see honesty notes) |
| Qwen3-8B Q4_K_M | 5.03 GB | 32 s | 2.3 s | 39.55 | 758.08 |
| Qwen3-14B Q4_K_M | 9.00 GB | 32 s | 3.9 s | 23.21 | 481.89 |
| gpt-oss-20b MXFP4 | 12.11 GB | 51 s | 57.3 s | **59.21** | 1568.00 |

**The headline: gpt-oss-20b runs on the free T4.** The 12.11 GB file fits the 15360 MB card with room for context, every layer offloads, and it decodes at 59.21 tok/s — because the MoE only activates ~3.6B params per token, it is 2.5× faster than the 14B dense model. Nobody needs to guess anymore.

**Honesty notes** (why there is no asterisk-free way to publish benchmarks):

- The 4B model's prompt-processing number was a harness artifact (it violated the card's memory bandwidth) and is excluded; the other three are consistent with each other and with physics.
- `nvidia-smi memory.used` under-reports llama.cpp's footprint on this driver (~55–75% of the file size at idle). File size is the real number; the reported figures are only good for relative comparison. llama.cpp's own `offloaded N/N layers` lines are in the committed kernel log.
- gpt-oss-20b ran without its harmony chat template applying (raw `<|channel|>` markers in the output). Decode speed is unaffected; expect to fight the template in llama-cpp-python.

**The build gotcha that cost a kernel run:** `pip install llama-cpp-python` from source dies on the Kaggle image before compiling anything (nvcc 12.8 is present, so the failure is elsewhere — the truncated log hid it). Don't debug it; point pip at the project's own prebuilt CUDA wheel index (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`): 65.8 s to full GPU offload. Three kernel versions in `results/` tell the whole story.

### Eval sibling: [recommend-quant-15gb](https://www.kaggle.com/benchmarks/tasks/emdadh/recommend-quant-15gb/2)

Hosted models were asked to advise on running models on a 15360 MB card — graded against **our measured table**, not a textbook answer key. Deterministic grading, no LLM judge.

| Model | Score | $/item | out tok/item |
|---|---|---|---|
| `gemini-3.5-flash-lite` | 4/5 | $0.0000912 | 28.2 |
| `gemini-3.7-flash` | **5/5** | $0.00207225 | 538.8 |
| `gpt-oss-20b` | **5/5** | $0.0000912684 | 341.8 |

14/15. The cheap model lost its perfect score to the one item that is pure arithmetic (picking the largest file that fits with headroom — it answered conservatively and wrongly). The expensive model bought the same 5/5 as gpt-oss-20b at **22.7× the price per item**. And one failure was ours: the first grader rejected gpt-oss-20b's legitimately correct lowercase `q4_0` on case-sensitive matching — fixed and re-run, because a grader that fails a right answer is a broken grader.

## Month 1 — calibration (kept for the receipts)

## W0/1 results — calibration

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

Evidence: `projects/2026-W38-hello-t4/results/{timings.json,eval_results.json}` (committed — every digit above is checkable in-repo).

## W2 results — the three-lane bake-off (calibration)

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

Evidence: `projects/2026-W38-bakeoff-t4/results/` (committed — raw per-item run files included).

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

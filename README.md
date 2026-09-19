# Free GPU Lab

Unattended measurements on **Kaggle GPU** (~30h/week) and **Kaggle Benchmarks** ($10/day, $100/month Model Proxy), with Colab as the public notebook surface.

W0/1 GPU lane is done. Proxy lane is prepared but unspent.

| Meter | Live (2026-09-19) |
|---|---|
| Kaggle GPU | **0.01 / 30h** (refresh 2026-09-26) |
| Kaggle TPU | 0 / 20h |
| Benchmarks Proxy | $0.00 spent — $10/day and $100/month unused |
| Colab | no active sessions |

### Latest measured result — W0/1 Hello T4

Free Kaggle T4, harvested with no browser. Kernel: [emdadh/hello-t4](https://www.kaggle.com/code/emdadh/hello-t4)

| Field | Measured |
|---|---|
| GPU | Tesla T4 (T4 stuck, not the P100 fallback) |
| VRAM | 15360 MB |
| fp16 matmul | 21162.1 GFLOP/s |
| CUDA / driver | 13.0 / 580.159.04 |
| pip install | 4.39 s |

Every number traces to `projects/2026-W38-hello-t4/results/timings.json`.

Plan: [PLAN.md](PLAN.md). Agent rules: [AGENTS.md](AGENTS.md). Paused-state notes: [HANDOFF.md](HANDOFF.md).

```bash
source .venv/bin/activate       # Python 3.12 + kaggle 2.2.4
pytest                          # 48 tests
.venv/bin/kaggle quota          # GPU/TPU hours
.venv/bin/python -m lab.task_check   # free pre-flight for a Benchmarks task file
```

Kaggle user: `emdadh`. CLI binary: `.venv/bin/kaggle` — not the system 1.7.4.5 shim.
Colab CLI: `~/.local/bin/colab` (google-colab-cli 0.6.0, auth via `oauth2`).

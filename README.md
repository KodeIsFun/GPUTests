# Free GPU Lab

Unattended measurements on **Kaggle GPU** (~30h/week) and **Kaggle Benchmarks** ($10/day, $100/month Model Proxy), with Colab as the public notebook surface.

This week: Lab OS only. No GPU hours or Proxy dollars spent yet.

| Meter | Live (2026-09-19) |
|---|---|
| Kaggle GPU | 0 / 30h (refresh 2026-09-26) |
| Kaggle TPU | 0 / 20h |
| Benchmarks Proxy | $10/day and $100/month unused |

Plan: [PLAN.md](PLAN.md). Agent rules: [AGENTS.md](AGENTS.md).

```bash
source .venv/bin/activate   # Python 3.12 + kaggle 2.2.4
pytest
.venv/bin/kaggle quota      # GPU/TPU hours
```

Kaggle user: `emdadh`. CLI binary: `.venv/bin/kaggle` — not the system 1.7.4.5 shim.

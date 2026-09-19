# Free GPU Lab — how the agent operates

Use `.venv/bin/kaggle` (2.2.4). Never the pyenv 1.7.4.5 shim.

## Hard stops

- Kaggle GPU: max **8h** per job, **24h** used per week (leave 6h of the 30h).
- Benchmarks Proxy: max **$8/day**, **$90/month**. Default **$3/day**. Two models per `kaggle b t run`.
- Before a GPU push: `kaggle quota` → `lab.quota.decide_gpu_job`.
- Before a Proxy run: remaining daily/monthly vs `estimated_usd`. CLI 2.2.4 has no `kaggle b quota`; use the screenshot meters + `ledger/proxy.jsonl`.
- Tweets: every number must appear in `timings.json` or `eval_results.json`.
- Task files: `@kbench.task(name=slug)` **and** `.run()` / `.evaluate()`. Missing `.run()` is a silent no-op.
- Colab: `colab run` (no `--keep`). Always `colab stop`. Never `auth` / `drivemount` / `repl`.
- Proxy is eval-only. No tweetauto, no video pipeline, no app backend.

## Auth

`~/.kaggle/access_token` is enough. Do not demand `kaggle.json`.

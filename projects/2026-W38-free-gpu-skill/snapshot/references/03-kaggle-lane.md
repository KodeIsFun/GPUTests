# 03 — Kaggle lane (unattended measurements)

Kaggle is the lane for **repeatable, unattended, harvestable** runs: a pushed
kernel executes on their T4 without anyone watching, and you pull the results
file afterwards. It cannot serve APIs (no inbound connections) — that is the
Colab notebook's job. One-time human setup: Path C in
[01-manual-steps.md](01-manual-steps.md).

## Install + auth (agent)

```bash
python3.11 -m venv .venv && . .venv/bin/activate   # CLI 2.2 needs Python ≥3.11
pip install -U "kaggle>=2.2"
# put the copied token where the CLI looks:
printf '%s' "<TOKEN>" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
kaggle kernels list -m --page-size 1   # smoke: must NOT raise "could not find kaggle.json"
kaggle quota                           # GPU row: ~30h/week, 8h/job cap
```

## Push the benchmark job

`templates/kaggle-job.py` is the measured W5 job (v3 protocol): per model it
times download, load, prompt-512 on a clean context, and greedy decode; every
stage is flushed to `timings.json` so a timeout still leaves partial results;
VRAM is checked with honest margins and an OOM is recorded as a result.

1. Copy the template into a folder next to a `kernel-metadata.json`:

```json
{
  "id": "<kaggle-username>/llm-bench",
  "title": "LLM bench",
  "code_file": "job.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [], "kernel_sources": [],
  "competition_sources": [], "model_sources": []
}
```

2. Edit `MODELS` in `job.py` (catalog with verified ungated repos is at the
   top of the file) and set `GIT_SHA` (any provenance string).
3. Push with a **binding time limit**:

```bash
kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4 -t 7200
```

`-t` seconds is enforced by the platform; keep it ≥2× your estimate.

4. Poll and harvest:

```bash
kaggle kernels status <username>/llm-bench   # RUNNING → COMPLETE
kaggle kernels output  <username>/llm-bench -p out/ --force
```

`--force` matters: the CLI skips the download when a local file is newer.

## Rules this lane enforces on you

- **Kaggle derives the kernel URL-slug from the TITLE.** If
  `slugify(title) != metadata id`, the first push succeeds under the
  title-derived slug and every later push 409s. Keep them identical.
- **The delivered card is whatever the pool has.** Requesting T4 works today;
  requesting P100 silently delivers a T4 (the P100 pool is gone). Never trust
  the flag — read `gpu` from `timings.json`.
- A full four-model sweep costs ~5 minutes of GPU and downloads ~28 GB of
  weights in under 3 minutes on their network. The 30 h/week allowance is
  effectively unlimited for this workload — the risk is *hanging*, not cost.
- If a run hangs: `kaggle kernels status` shows RUNNING until the `-t` cap
  kills it. The incremental `timings.json` means the partial table is still
  harvestable. Do not re-push blindly; read what landed first.

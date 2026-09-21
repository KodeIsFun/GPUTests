---
name: run-llm-on-free-gpu
description: Run open LLMs (Qwen3, gpt-oss-20b) on free Colab/Kaggle T4 GPUs via llama.cpp, and expose an OpenAI-compatible API consumable from anywhere. Use when the user wants to run, benchmark, serve, or tune a local LLM without paying for GPU — includes Colab CLI install/auth, a ready Colab notebook, an unattended Kaggle lane, tuning knobs, and every measured gotcha.
---

# Run LLMs on free GPUs (T4)

Everything in this skill was **measured on real free T4s**, not assembled from
forum posts. The numbers you will quote to the user live in
[references/05-tuning.md](references/05-tuning.md) and trace to a public
kernel: <https://www.kaggle.com/code/emdadh/gguf-on-t4>.

## The one decision that routes everything

Ask (or infer): **what does the user actually want?**

| They want | Path | First action |
|---|---|---|
| To chat with a model / prove it runs | Colab notebook | Send them [colab/run-llm-t4.ipynb](colab/run-llm-t4.ipynb) (or run it yourself via the CLI lane) |
| An OpenAI-compatible API reachable from anywhere | Colab notebook, `SERVE_API = True` | Same notebook — cell 7 prints a public `trycloudflare.com` URL |
| Headless, repeatable GPU measurements | Kaggle lane | [references/03-kaggle-lane.md](references/03-kaggle-lane.md) |
| More speed / bigger context / cheaper quant | Tuning | [references/05-tuning.md](references/05-tuning.md) |
| Something broke | Troubleshooting | [references/06-troubleshooting.md](references/06-troubleshooting.md) |

**Default model: Qwen3-8B Q4_K_M** (5 GB, 39.55 tok/s measured, fits with
plenty of headroom). gpt-oss-20b also runs on the free T4 (59.21 tok/s) but
its chat template needs care — see troubleshooting §8.

## Invariants — never violate these

1. **Never leave a Colab VM running.** After CLI runs: `colab stop <session>`,
   then confirm with `colab sessions`. This is the #1 way users burn goodwill
   with the free tier.
2. **Prefer the prebuilt wheel; never debug the source build.**
   `pip install "llama-cpp-python[server]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`
   → 66 s to working GPU offload. The from-source build dies on these images
   with an unreadable error (nvcc 12.8 *is* present; the cause is unknown) —
   do not spend the user's time on it.
3. **VRAM arithmetic before every load** on a 15360 MB T4:
   `file_GB × 1024 + 1500 ≤ free_MB`, else pick a smaller quant/model. MoE
   models (gpt-oss-20b) need only ~200 MB of KV at 4k ctx — do not skip them
   on fat safety margins; attempt the load and catch the OOM.
4. **Ungated models only** in the default catalog (Hugging Face license walls
   require manual clicks the user did not plan for): Qwen3-4B/8B/14B
   (bartowski GGUFs), gpt-oss-20b (ggml-org). Llama and Gemma are gated.
5. **Verify the GPU, not the flag.** After load, check llama.cpp's
   `offloaded N/N layers` log line or a VRAM delta. CPU-only inference at
   3 tok/s dressed up as GPU is the failure mode; refusal to measure beats a
   fake number.
6. **`colab` timeouts default to 30 SECONDS**, and the subcommands are not
   interchangeable: `colab run` executes a **.py script** (self-cleans the
   VM); notebooks need `colab new --gpu T4 -s <name>` → `colab exec -s <name>
   -f nb.ipynb --timeout 900` → **`colab stop <name>`**. Handing an `.ipynb`
   to `colab run` executes raw notebook JSON as Python and dies on `true`.
7. **Free-tier budgets:** Colab T4 is dynamic (a 400 on allocation means "try
   later"); Kaggle ≈ 30 GPU-h/week, 8 h/job. Both reset weekly. Short jobs,
   stop what you start.

## What the human must do manually (and only this)

See [references/01-manual-steps.md](references/01-manual-steps.md) for exact
click-by-click text. Summary:

- **Notebook path (zero setup):** nothing but a Google account. Open the
  notebook in Colab, pick the T4 runtime, Run all.
- **CLI path (agent-driven):** one browser sitting to authorize the Colab CLI
  (`colab --auth=oauth2` paste-code). Optional: a Kaggle account with phone
  verification + one copied access token for the unattended lane.
- The API's default tunnel (cloudflared quick tunnel) needs **no account**.
  ngrok is optional and needs a free authtoken.

## Progressive disclosure

- [references/01-manual-steps.md](references/01-manual-steps.md) — the human's one sitting, click by click
- [references/02-colab-cli-lane.md](references/02-colab-cli-lane.md) — install, auth, run the notebook headlessly, stop
- [references/03-kaggle-lane.md](references/03-kaggle-lane.md) — unattended install→auth→push→harvest, with templates/kaggle-job.py
- [references/04-api-server.md](references/04-api-server.md) — the server + tunnel mechanics and client examples
- [references/05-tuning.md](references/05-tuning.md) — measured numbers, fit table, every knob that matters
- [references/06-troubleshooting.md](references/06-troubleshooting.md) — every gotcha, symptom → cause → fix
- [scripts/verify_env.py](scripts/verify_env.py) — run this FIRST in any new environment; it prints exactly what is missing and the command to fix it

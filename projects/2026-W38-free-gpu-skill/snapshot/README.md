# run-llm-on-free-gpu

Run real open-weights LLMs on **free** Colab/Kaggle T4 GPUs with llama.cpp —
and expose them as an **OpenAI-compatible API consumable from anywhere**.
Every number in this repo was measured on an actual free T4, and every
gotcha in it was paid for in a real failed run. Nothing is aspirational.

Written to be **executed by an agent** (any coding agent, no special tools or
MCPs required — only Bash + Python) while keeping the human's manual part to
a single short sitting. If you are an agent: read
[SKILL.md](SKILL.md) first and follow its routing table.

## What you get

| Path | What | Human effort |
|---|---|---|
| **Notebook** ([colab/run-llm-t4.ipynb](colab/run-llm-t4.ipynb)) | open on Colab, pick T4, Run all → install, download an ungated model, GPU-offload proof, chat demo, and a public API URL (Cloudflare quick tunnel — no account) | ~2 min |
| **Colab CLI** | the same notebook run headlessly by an agent (`colab run`) | one OAuth consent |
| **Kaggle lane** | the same measurement as a pushed, unattended kernel with harvested `timings.json` | account + phone verify + one token copy |

## The measured table (free Kaggle T4, llama.cpp, 4k ctx, greedy)

| Model | File | Decode | Prompt-512 |
|---|---|---|---|
| Qwen3-4B Q4_K_M | 2.50 GB | 62.64 tok/s | n/a¹ |
| Qwen3-8B Q4_K_M | 5.03 GB | 39.55 tok/s | 758.08 tok/s |
| Qwen3-14B Q4_K_M | 9.00 GB | 23.21 tok/s | 481.89 tok/s |
| gpt-oss-20b MXFP4 | 12.11 GB | **59.21 tok/s** | 1568.00 tok/s |

¹ excluded — the measurement beat the card's physics; we don't publish
numbers we can't defend.

**The headline:** gpt-oss-20b — the open model everyone wants — runs on a
free T4: no OOM, 59.21 tok/s, 2.5× faster than the 14B dense model, because
the MoE only activates ~3.6B params per token.

Source kernel (exact code + logs): <https://www.kaggle.com/code/emdadh/gguf-on-t4>

## Repo map

```
SKILL.md                    ← agents start here (routing + invariants)
colab/run-llm-t4.ipynb      ← THE notebook (verified end-to-end on a free T4)
references/
  01-manual-steps.md        ← everything the human must click, click-by-click
  02-colab-cli-lane.md      ← install/auth/run/stop, headless
  03-kaggle-lane.md         ← unattended push → harvest, with the metadata trap
  04-api-server.md          ← server + tunnel mechanics, client examples
  05-tuning.md              ← measured numbers, fit table, knobs in impact order
  06-troubleshooting.md     ← 13 gotchas, symptom → cause → fix
scripts/verify_env.py       ← agent preflight: what's installed, what's authed,
                              what manual step is missing (with exact commands)
templates/kaggle-job.py     ← the measured benchmark job, self-contained
tools/build_notebook.py     ← regenerates the notebook from source (edit here)
```

## Quickstart (human, no agent involved)

1. Open [colab/run-llm-t4.ipynb](colab/run-llm-t4.ipynb) in
   [Colab](https://colab.research.google.com).
2. `Runtime → Change runtime type → T4 GPU → Save`.
3. `Runtime → Run all`.
4. ~3–5 minutes later: a chat answer, and a `trycloudflare.com` URL that
   serves an OpenAI-compatible endpoint of your model from anywhere.

## Honest limits

- The API lives exactly as long as the Colab session (~90 min idle limit);
  it is a working endpoint, not a deployment. URLs are ephemeral by design.
- The free T4 is dynamic: allocation can 400 ("try later"). The notebook
  degrades to CPU with a clear warning rather than pretending.
- gpt-oss-20b's harmony chat template does not auto-apply through
  llama-cpp-python — decode speed is unaffected, but expect channel markers
  in raw output (troubleshooting §8).
- Numbers move with driver/library updates. Re-run the Kaggle lane template
  to re-measure before quoting anything as current.

## Provenance

All measurements and gotchas come from the Free GPU Lab
(<https://github.com/KodeIsFun/GPUTests>) — a lab that runs on free tiers
only and commits the evidence next to every claim. The benchmark job in
`templates/` is the same code that produced the table above.

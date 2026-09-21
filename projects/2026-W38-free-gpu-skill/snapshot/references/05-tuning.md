# 05 — Measured numbers and the knobs that matter

Everything here was measured on a **free Kaggle T4 (15360 MB, driver 580.x,
CUDA 12.8)** with llama-cpp-python 0.3.35 (prebuilt cu124 wheel), 4k context,
greedy decoding, all layers on GPU. Source of truth (public kernel with the
exact code and logs): <https://www.kaggle.com/code/emdadh/gguf-on-t4>.
Reproduce it headlessly with [03-kaggle-lane.md](03-kaggle-lane.md).

## The table

| Model | File | Load | Decode (tok/s) | Prompt-512 (tok/s) |
|---|---|---|---|---|
| Qwen3-4B Q4_K_M | 2.50 GB | 2.0 s | **62.64** | n/a¹ |
| Qwen3-8B Q4_K_M | 5.03 GB | 2.3 s | 39.55 | 758.08 |
| Qwen3-14B Q4_K_M | 9.00 GB | 3.9 s | 23.21 | 481.89 |
| gpt-oss-20b MXFP4 | 12.11 GB | 57.3 s | **59.21** | 1568.00 |

¹ The 4B prompt-processing number measured faster than the card's physics
allow — a harness artifact, excluded on purpose. Do not quote one for it.

How to read this for a user:

- **Decode speed** is what streaming feels like. 20+ tok/s reads comfortably;
  below ~10 feels sluggish.
- **Prompt-512** is time-to-first-token for big inputs (RAG, long history):
  512 tokens at 482 tok/s ≈ 1.1 s before the first word.
- **gpt-oss-20b is the surprise**: MoE (~3.6B active params) nearly matches
  the 4B dense model and beats the 14B by 2.5×. If the user wants "the
  biggest brain that's still fast", it is the answer on a free T4 — with the
  template caveat (troubleshooting §8).
- Rule of thumb: decode speed ≈ 320 GB/s ÷ file size, times ~0.6 real-world
  efficiency. Sanity-check any number you're about to promise against this.

## Model fit on 15360 MB (Q4_K_M unless noted)

| Fits comfortably | Fits with care | Does not fit |
|---|---|---|
| ≤ 9 GB files (Qwen3-14B Q4: 9.0 GB) | 12.1 GB (gpt-oss-20b: ~2.9 GB spare for KV+overhead) | anything ~13 GB+ (70B Q4 ≈ 40 GB — never) |

The precheck the notebook runs: `file_GB × 1024 + 1500 ≤ free_MB`. The 1500 MB
covers KV (4k ctx ≈ 200 MB on 14B, less on MoE) plus CUDA context and
buffers. Tighten only if you measured; fat margins caused a real bug where
gpt-oss-20b was skipped without being attempted.

## Knobs, in order of impact

1. **Quant level = the only lever that changes which models fit.** Q4_K_M is
   the default sweet spot (quality-per-GB). Go Q4_K_S / Q3_K_M to squeeze a
   bigger model in; never Q8 on ≥8B models at 15 GB — you'll spend all headroom
   on weights and OOM at the first long prompt.
2. **`n_ctx`**: every extra 4k of context costs ~0.2 GB (dense 14B) — keep 4096
   unless the user needs long documents; then enable KV quantization before
   raising it (next knob).
3. **KV cache quantization**: `--cache-type-k q8_0 --cache-type-v q8_0` halves
   KV size at negligible quality cost — the cheapest way to buy context.
4. **`flash_attn=True`** (server: `--flash-attn`): try it and verify output
   sanity; gains are small on T4 (sm_75) but free when they land.
5. **`n_gpu_layers`**: always `-1` here; a partial offload is strictly worse
   (PCIe round-trips). If a model doesn't fully fit, the fix is a smaller
   quant, not partial offload.
6. **`n_threads`**: the GPU does the work; leave Python-side threads at
   default (2–4). More threads on these 2-vCPU VMs just adds contention.
7. **Temperature/greedy**: benchmarks and reproducibility → temperature 0.
   Chat → 0.7. Never benchmark with sampling on and wonder why numbers move.

## What NOT to bother with

- Source builds with custom flags — the prebuilt wheel is what works (§02).
- Double-checking "P100 vs T4": the free pool hands out T4s; there is no
  P100 anymore, and the accelerator flag never changed that.
- `nvidia-smi memory.used` as a VRAM-truth source: it under-reports llama.cpp
  on these drivers (~55–75% of file size). Use file size + the offload log.

# W40 — Pruna vs Viggle head-to-head on a free T4 (Qwen-Image-2.1)

Experiment 2 of the W39/W40 scouting backlog: two competing distillations of
Qwen-Image-2.1, released 3 days apart (Pruna 2026-09-24, Viggle v0.2.1
2026-09-24), no published T4 numbers for either. Same card, same canary, same
seed — the lab's comparative format.

## Fair-comparison design (recorded in timings JSON)

- **Same LoRA loader for both sides**: the viggle author's runtime-hook node
  (`ViggleTurboLora`, tooltip: "diffusers-format Qwen-Image-2.1 LoRA").
  Verified compatible *before* the run by reading both safetensors headers via
  HTTP range request: identical `transformer.` diffusers key layout, Pruna
  rank 64 (alpha 128) vs Viggle rank 256, same fused-SwiGLU `img_mlp` pattern.
  Hooks keep the LoRA update exact on quantized weights; stock merge is lossy
  per the node's own docstring. Zero `lora key not loaded` warnings, zero
  fallbacks in the final run.
- **Each recipe's own published sigma schedule**: Viggle = resolution-shifted
  6-step (their node shifts mu by latent tokens); Pruna = raw card sigmas, no
  shift (`1.0, 0.94, 6/7, 2/3, 0.4` for 5 steps; 8-step adds `14/15, 10/13,
  6/11, 2/9`) — matched exactly against the `inference_sigmas` field stashed
  from Pruna's own safetensors metadata. Carried by a 12-line local node
  (`pruna_sigmas.py`, written by `setup()`).
- **Same base weights**: unsloth GGUF Q4_K_M unet + abenzerps int8 TE + bf16
  VAE, `--force-fp16 --disable-comfy-compiler`, cfg off (both cards prescribe
  true_cfg_scale=1.0), euler, seed 42, the two v4/v5 canary prompts.

## Results (final run: fresh Colab T4 VM, `results/timings_v6_pruna.json`)

| row | wall | s/step |
|---|---|---|
| pruna5 cat 864x576 (cold, incl. one-time load) | 69.1 s | 2.43 |
| **pruna5 sign 768x768 warm — headline** | **21.0 s** | 2.90 |
| pruna8 sign 768x768 warm (card's quality default) | 30.0 s | 3.04 |
| pruna5 sign 1024x1024 warm (Pruna's training res) | 33.0 s | 5.61 |
| viggle r256 sign 768x768 6s warm (in-session anchor) | 33.0 s | 3.87 |
| viggle r256 sign 1024x1024 6s warm | 42.0 s | 6.27 |

Headline: Pruna five-step 21.0 s vs Viggle six-step 33.0 s (1.57x) at 768²;
Pruna's own eight-step quality mode costs 30.0 s — the same bill as Viggle.
At 1024²: 33.0 s vs 42.0 s (1.27x). Canaries: "FREE GPU LAB" letter-perfect
on both distillations at 768²; at 1K Pruna five-step shows minor ghost
sub-text around a correct main sign, Viggle stays perfect. All six PNGs are
byte-identical (sha256) across two different VMs — same-seed sampling is
deterministic on this stack. v5 cross-check: the viggle anchor ran 33.0 s
in-session vs 24.0 s on v5's VM — cross-session wall comparisons need the
in-session anchor, which is why rows 5-6 exist.

Recipe difference recorded, not exercised: Pruna's card offers `kv_cache`
(diffusers-pipeline feature; no ComfyUI equivalent used here).

## Attempt history (why four runs)

1. `results-v6-attempt1/` — 7-row matrix repeated the headline graph with the
   same seed: ComfyUI returned a 3.0 s **execution-cache hit**, not a render.
   Fix: every row is now a distinct graph, asserted in the job.
2. `results-v6-attempt2/` — clean 6-row matrix, but the timings JSON labeled
   the surface `kaggle`: this Colab image ships stray `/kaggle` **and**
   `/kaggle/working` dirs, defeating path detection.
3. (log not archived) `/kaggle/working` detection also wrong — superseded.
4. `results/` — final: surface via env proof (`KAGGLE_KERNEL_RUN_TYPE` unset,
   `COLAB_GPU=1`, recorded as `surface_proof`), fresh VM, all rows valid.

## How it runs

`colab new -s w40-v6 --gpu T4` → `colab upload job_v6_pruna.py` → detached
launch → poll `/content/job.log` → `colab download` artifacts → **`colab
stop`**. Fresh-VM cost ≈ 9 min (14.5 GB weights + boot + 6 renders); with
weights cached ≈ 4 min.

## Tweets

`tweets/01-pruna-vs-viggle.txt` — gate PASS against
`results/timings_v6_pruna.json` (`lab.tweets.verify_from_files`). Unposted.

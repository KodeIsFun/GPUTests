# Plan: Viggle turbo LoRA probe — does the community 6-step recipe hold on free-tier weights?

**Question.** The 4-step shortcut failed quality (sign illegible, `results-v3/`). The community's own
answer to "fewer steps" is [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo):
a DMD2-distilled LoRA claiming ~5x faster than the 40-step base at "very competitive" quality, 6 steps,
CFG off. Does that recipe hold **on our free-tier setup** — GGUF Q4_K_M weights on a Colab T4 — and what
does a warm image cost at 6 steps vs the 12-step default (~79-87 s)?

## What the HF card actually specifies (decoded 2026-09-26 from the repo)

- LoRA v0.2.1, 6 steps, raw sigma nodes `1.0, 0.9375, 0.875, 0.75, 0.5, 0.25`, shifted per-resolution
  (dynamic exponential shift, mu from token count). **Not expressible in stock KSampler** — the author
  ships `comfyui/viggle_turbo.py` (ViggleTurboSigmas + ViggleTurboLora custom nodes) and an official
  `comfyui/Qwen-Image-2.1-viggle-turbo-t2i.json` workflow.
- CFG off entirely (`BasicGuider`, no negative pass): 6 forward passes per image vs 24 for our
  12-step cfg 2.5 default. Euler sampler.
- The LoRA must be applied at runtime, **never merged**: their node docstring — merging on bf16 keeps
  only ~70% of the update; on int8 weights requant noise is ~4x the update size. Their ComfyUI workflow
  uses the r128 file (679.6 MB) at strength 1.0; r256 (1.36 GB) is the diffusers-recommended file.
- Known gaps per card: complicated multi-reference edits, small/dense text (8 steps narrows it).
- License: Qwen Research, non-commercial. Fine for the eval lane, not for a product.

## Deviations from the author's recipe (and why)

| Author | Ours | Why |
|---|---|---|
| bf16 or int8_convrot diffusion weights | unsloth `qwen-image-2.1-Q4_K_M.gguf` (4.20 GB) | T4 16 GB free tier is the whole lab premise. Base (non-UC) restored: abenzerps restructured to UC-only, unsloth still hosts base — matches the LoRA's training distribution. Quant provenance differs from v2b's file (4.20 vs 4.60 GB), so an in-session 12-step anchor row measures the weight-swap effect. |
| TextEncodeQwenImage21 | CLIPLoader(type=qwen_image) + CLIPTextEncode | Proven in v2b/v3 lineage; sigma shift depends on latent size, not text encoding, so this doesn't touch the recipe. |
| EmptyLatentImage | EmptySD3LatentImage | v3-proven; the sigmas node handles both (`downscale_ratio_spacial` default 16). |

## Fixed settings

| Setting | Value |
|---|---|
| Diffusion weights | `qwen-image-2.1-Q4_K_M.gguf` (unsloth, base, 4.20 GB) |
| LoRA | `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors`, strength 1.0, via **ViggleTurboLora** (runtime hooks, no merge) |
| Text encoder / VAE | `qwen3vl_8b_int8_convrot.safetensors` / `qwen_image_2.1_vae_bf16.safetensors` (unchanged) |
| Flags | `--force-fp16 --disable-comfy-compiler` (unchanged) |
| Turbo sampler | euler + ViggleTurboSigmas `1.0, 0.9375, 0.875, 0.75, 0.5, 0.25`, BasicGuider (cfg off), SamplerCustomAdvanced |
| Anchor sampler | res_multistep / simple, 12 steps, cfg 2.5 (v3 recipe, unsloth weights) |
| Seed | 42 everywhere |

## Rows (in run order)

1. `turbo_cat_864x576` — 6-step turbo arm, cold (includes model load).
2. `turbo_sign_768x768` — 6-step turbo arm, warm. **The quality canary.**
3. `anchor_sign_12step_cfg2.5` — no LoRA, v3 recipe on unsloth weights. Bridges v2b/pocket numbers to the new weights.
4. `turbo_sign_8steps` — contingency, front-loaded: sigmas `1.0, 0.96875, 0.9375, 0.90625, 0.875, 0.75, 0.5, 0.25` (author's rule: add only at the high-noise end, keep 0.875/0.75/0.5/0.25). Only matters if row 2's text is illegible.
- Auto-fallback inside rows 1-2: if the ViggleTurboLora hook node errors on the GGUF-loaded model
  (card admits the ComfyUI port is "vibe-coded", verified only against diffusers), retry the row with
  stock `LoraLoaderModelOnly` (merge path, euler/simple 6 steps cfg 1.0 as approximation) and mark it.

## Expected numbers (hypothesis, to verify)

- Warm turbo render ≈ 6 passes x ~2.7-3.1 s + overhead ≈ **~20-30 s** (vs 79-87 s 12-step default).
  v3 measured 2.72 s/it at cfg-off 4-step on the same class of weights.
- Anchor ≈ 80-90 s warm (12 steps x 2 passes). If it drifts from 87.1 s, the unsloth-vs-UC quant swap
  moved speed — cross-run comparisons must then use the in-session anchor, not the old JSONs.
- Session ≈ 12-15 min including ~15.6 GB of downloads.

## Verdict criteria

- **Pass** → 6-step turbo becomes a second lab preset ("turbo") alongside the 12-step default: sign
  spells "FREE GPU LAB" legibly, cat keeps yellow raincoat + siamese look, no smearing at 100% zoom.
  Speed is measured, not assumed, from rows 1-3.
- **Partial** → quality holds on cat but sign text fails at 6 steps and reads correctly at 8: record
  both, preset becomes 8-step turbo for text prompts.
- **Fail on quality** (both step counts illegible/degraded) → turbo is closed for T4 GGUF; the
  failure mode on quantized weights is itself the result (the card's claims are bf16/int8 only).
- **Fail on error** (hook node broken on GGUF AND merge fallback degraded) → record which path died;
  diffusers-on-bf16 with CPU offload is out of scope for the T4 lane.

## Evidence rule

Every number in any tweet about this run must appear in `results-v4/timings_v4_turbo.json`.
Spelled-out numbers do not pass the gate — digits or nothing.

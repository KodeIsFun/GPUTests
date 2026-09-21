# 2026-W39 — Qwen-Image-2.1 GGUF Q4_K_M on a free Colab T4 (W6 image-gen probe)

Owner's prompt: "we want to test this in colab" →
<https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/blob/main/qwen-image-2.1-Q4_K_M.gguf>

Qwen-Image-2.1 is the successor to the 20B Qwen-Image: a **7B** diffusion
transformer (32 single-stream DiT layers), Qwen Research License. The GGUF
repo quantizes it and documents one runtime: ComfyUI + the **leejet** fork of
ComfyUI-GGUF (the city96 fork throws "Unknown model architecture!").

## Pieces

| File | Size | Role |
|---|---|---|
| `qwen-image-2.1-Q4_K_M.gguf` | 4.60 GB | diffusion model → VRAM |
| `qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB | text encoder → card says "RAM" |
| `qwen_image_2.1_vae_bf16.safetensors` | 676 MB | VAE |

The free Colab T4 has 16 GB VRAM but only **~12.7 GB system RAM**, so the
card-recommended "int8 encoder in RAM" combo is expected to be the failure
point. `job.py` runs an honest ladder and records every attempt:

1. card-recommended setup (int8 TE, default flags),
2. int8 TE + `--lowvram` (the card's own fallback advice),
3. smallest Comfy-Org text-encoder quant for 2.1, picked at runtime from the
   HF API (`Comfy-Org/Qwen-Image-2.1`, w4a8/int4 preferred).

OOM is a result, not a failure — the run only "fails" if no attempt produces
both images.

## Settings assumption (recorded in timings.json)

The 2.1 model card publishes no sampler settings; its diffusers example uses
40 steps @ 2048². We run **20 steps @ 1024², cfg 2.5, res_multistep/simple,
seed 42** — the cfg/sampler carried over from the Qwen-Image 1.0 Comfy-Org
template. Any quality verdict must be read against that downscale.

## How it runs

`colab new -s w6-qwen21 --gpu T4` → `colab upload job.py` → detached launch →
poll `/content/job.log` → `colab download` artifacts → **`colab stop`**.

`job.py` is self-diagnosing: before any attempt it verifies node names against
ComfyUI's `/object_info` (UnetLoaderGGUF*, CLIPLoader `qwen_image` type,
EmptySD3LatentImage, `res_multistep`) so a node-rename fails in 5 s with the
actual enum list in the log, not after a 30-min image.

Prompts (fixed, recorded in timings.json):
1. `p1_cat_raincoat` — photographic.
2. `p2_freegpulab_sign` — neon sign reading "FREE GPU LAB", mirroring the
   official card's own text-rendering example style.

## Results

`results/` — filled after the run: `timings.json`, images, ComfyUI logs,
`job.log`. Tweet drafts go in `tweets/` only after gate verification against
`timings.json`.

---

## Result (2026-09-21, run w6-qwen21, T4, torch 2.11.0+cu128)

**Success on attempt 1 — the card-recommended setup runs on a free Colab T4
with no flags.** Both images generated at 1024×1024, 20 steps.

| Metric | Value | Source |
|---|---|---|
| Image 1 cold (load + encode + 20 steps + decode) | **825.6 s** | timings.json |
| Image 2 warm (models resident) | **765.6 s** | timings.json |
| Sampling rate | **38.0 s/step** (20/20 bar: 12:36) | comfyui-a1.log |
| Peak VRAM | **14039 MB / 15360 MB (91%)** | timings.json |
| Peak RAM | **9671 MB / 12975 MB (75%)** | timings.json |
| Weight downloads (14.6 GB total) | 47.0 + 97.9 + 8.5 s @ ~96 MB/s | timings.json |
| Total wall (install→done) | 2097 s | timings.json |

The two fears that did not materialize:

- **RAM did not OOM.** ComfyUI's *dynamic VRAM loading* staged the 9.35 GB
  int8 encoder ("8916MB Staged") instead of materializing it — peak RAM 9.7 GB
  on a 12.7 GB box. The `--lowvram` and w4a8 ladder steps were never needed.
- **The sd.cpp-flavored GGUF loads fine** in the leejet fork:
  `compatibility mode 'sd.cpp' [arch:qwen_image21]`, qtypes BF16/Q4_K/Q6_K,
  `4487.25 MB loaded, full load: True`.

The real bottleneck is compute, not memory: the T4 has no bf16, so ComfyUI
runs the DiT with `weight dtype torch.bfloat16, manual cast: torch.float32`
→ **38 s/step** at 1024². An fp16-forced run (UnetLoaderGGUFAdvanced
`weight_dtype` or `--force-fp16`) is the obvious next experiment — Turing
fp16 tensor cores could plausibly halve it. For scale: SD1.5 does a 512²
step in ~0.3 s on the same card; the 7B DiT is ~100× slower per step at 4×
the pixels.

Image quality (eyeballed from downloaded PNGs, both committed):
`p1_cat_raincoat` is photorealistic and prompt-faithful;
`p2_freegpulab_sign` renders **"FREE GPU LAB"** spelled perfectly in neon
with rain reflections — the 2.1 text-rendering trick survives Q4_K_M
quantization at 20 steps.

Files: `timings.json` (single source of truth), `comfyui-a1.log`,
`job.log`, both PNGs. Session `w6-qwen21` stopped; nothing left running.

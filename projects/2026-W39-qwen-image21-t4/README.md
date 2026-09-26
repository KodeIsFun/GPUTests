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

---

## Result v2 (2026-09-21, sessions w6-qwen21-v2 / -v2c): the speed combo

Combo: **fp16 cast + 12 steps + 768²**, same seed and prompts as v1.

| Config | s/step | Cold image | Warm image | Source |
|---|---|---|---|---|
| v1: 1024², 20 steps, fp32 (bf16 model, T4 has no bf16) | 38.0 | 825.6 s | 765.6 s | `results/timings.json` |
| v2 control: 768², 12 steps, fp32 | — | 342.6 s | — | poller capture (see incident note) |
| **v2b combo: 768², 12 steps, fp16 + `--disable-comfy-compiler`** | **6.20** | **127.9 s** | **79.2 s** | `results/timings-v2b.json` |

**9.7× faster warm-image generation** (765.6 → 79.2 s). Attribution from
s/step: 6.13× total = ~2.96× settings (20→12 steps × 1024²→768²) × ~2.07×
fp16 — the fp16 win landed even better than the predicted 1.5–2×.

Quality (eyeballed from committed PNGs): fp16 introduces no visible artifacts;
the neon sign still reads **FREE GPU LAB** spelled perfectly at 12 steps.
Minor banding in the darkest reflection gradients, plausibly fp16 and/or
12-step related — judged acceptable for social content.

### The fp16 blocker and its fix (the gotcha worth posting)

`--force-fp16` alone **crashes on the T4**: the model loads fine
(`weight dtype torch.float16`) but the first forward dies with
`RuntimeError: aimdo memory compile error` from `comfy_aimdo/malloc_graph.py`
— ComfyUI's model-compiler/CUDA-graph subfeature can't compile the fp16 path
on this card. Adding **`--disable-comfy-compiler`** fixes it. The leejet
fork's `UnetLoaderGGUFAdvanced` exposes **no `weight_dtype` input at all**
(empty via `/object_info`), so the per-node fp16 route does not exist here;
the server flag is the only path.

### Incident note (kept honest)

The first v2/v2b artifact set was lost: artifacts were never downloaded (a
zsh word-splitting bug in the download loop), and the VM stop was bundled
into the same command. The v2 numbers and error text above survive via the
committed poller captures (`results/v2-poller-capture.log`,
`v2b-poller-capture.log`); the fp16 rows were re-generated cleanly on session
`w6-qwen21-v2c` (`timings-v2b.json`, both PNGs, `comfyui-v2b.log` are from
that run — numbers reproduced: 6.81 vs 6.20 s/step across the two runs).
Rule added to HANDOFF: verify every download before stopping a session, and
never bundle downloads with the stop.

Session stopped; nothing left running.

## 4-step follow-up (v3, 2026-09-24) — FAIL on quality, 12 steps stays default

Plan in `PLAN-4steps.md`; driver `job_v3_steps4.py`; artifacts in `results-v3/`
(`timings_v3.json`, `comparison_v3.png`, sign zooms). Same direct pipeline (no
Pocket rewriter), 4 steps instead of 12, everything else held fixed: fp16 +
`--disable-comfy-compiler`, cfg 2.5, `res_multistep/simple`, seed 42.

**Baseline deviation, forced upstream:** the non-UC `qwen-image-2.1-Q4_K_M.gguf`
now 404s — abenzerps restructured `Qwen-Image-2.1-GGUF` into
`Qwen-Image-2.1-Uncensored-GGUF` and dropped the non-UC diffusion file (TE/VAE
still resolve via redirect). So v2b is no longer a valid same-weights baseline;
the run switched to **UC-Q4_K_M** and compares against the Pocket probe's
DIRECT arm (same UC weights, 12 steps, seed 42): sign 768² 87.1 s, cat
864×576 126.1 s cold. Timings sizes were matched per prompt for an exact
comparison. (Also: the first launch attempt downloaded a 15-byte 404 body
because the job's wget path didn't hard-fail on rc — relaunch covered it;
husks left on the ephemeral VM.)

| Row | Result | vs 12-step UC baseline |
|---|---|---|
| cat 864×576, cfg 2.5 | 78.1 s (cold, includes model load) | 12-step cat was 126.1 s cold |
| sign 768×768, cfg 2.5 | **27.0 s** (25.29 s executed) | 87.1 s → **3.2× faster** |
| sign 768×768, cfg 1.0 (contingency) | 15.0 s | — |

**Verdict: quality fails, speed is real.** At cfg 2.5 the photographic prompt
survives (yellow raincoat intact, composition usable) but the text canary
collapses: the sign is dark, smeared, and illegible. The front-loaded cfg 1.0
contingency partially rescues it — "FREE GPU" becomes readable but the third
word degrades and letters wobble with ghost strokes; still far from the crisp
12-step render. Per the plan's criteria: 4 steps is closed, cfg-off does not
rescue it, 12 steps stays the default. The failure mode is specifically text
rendering; photographic content is the marginal-use case.

---

## Result v4 (2026-09-26, session w39-turbo): Viggle turbo LoRA — 6 steps PASS, new fast preset

Plan in `PLAN-turbo.md`; driver `job_v4_turbo.py`; artifacts in `results-v4/`
(`timings_v4_turbo.json`, `comparison_v4.png`, `comfyui-v4.log`, 4 PNGs). The
community answer to the 4-step failure: [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo),
a DMD2-distilled LoRA (v0.2.1) that claims 6 steps at "very competitive"
quality with CFG fully off. We ran their exact ComfyUI recipe on free-tier
weights.

**Verdict: PASS on both criteria — 6-step turbo becomes the lab's fast preset
for Qwen-Image-2.1 on T4.** The text canary that killed the naive 4-step probe
spells **FREE GPU LAB** perfectly at 6 steps (crisp neon, correct reflections),
and the photographic prompt keeps every element (siamese points, yellow hooded
raincoat, wet cobblestones, neon bokeh).

| Row (seed 42) | Wall | Executed | s/step |
|---|---|---|---|
| turbo cat 864×576, 6 steps (cold, incl. model load) | 72.1 s | 70.13 s | 2.34 |
| **turbo sign 768×768, 6 steps (warm)** | **21.0 s** | 20.17 s | 2.68 |
| anchor sign, no LoRA, 12 steps cfg 2.5 (warm) | 69.1 s | 67.79 s | 5.29 |
| turbo sign 768×768, 8 steps (contingency, not needed) | 30.0 s | 28.50 s | 3.13 |

- **3.3× vs the in-session anchor** (21.0 s vs 69.1 s, same VM, same unsloth
  weights) — the only clean comparison, since the old baselines used different
  weight files. Against the retired defaults: 87.1 s (UC, pocket probe) → 4.1×;
  79.2 s (v2b non-UC) → 3.8×.
- CFG-off halves per-step cost exactly as predicted: 2.68 vs 5.29 s/step
  (6 forward passes per image vs 24).
- The 8-step contingency (add high-noise sigma steps only, per the card) was
  never needed for this prompt class but is measured: 30.0 s.

### What the recipe actually is (and what we kept/changed)

The card's few-step claim is **not** a sampler tweak — it needs their custom
nodes (`comfyui/viggle_turbo.py`, shipped with the repo): a resolution-shifted
sigma schedule (`1.0, 0.9375, 0.875, 0.75, 0.5, 0.25`, dynamic exponential
shift from token count) that stock KSampler cannot express, and a **runtime
hook** LoRA loader because merging the LoRA is lossy (their docstring: ~70% of
the update kept on bf16, ~4x requant noise on int8). We kept their node, their
r128 LoRA at strength 1.0, euler, BasicGuider (no negative pass), EmptySD3LatentImage,
and swapped only the weights to **unsloth base (non-UC) Q4_K_M** — abenzerps'
repo is UC-only since the restructure, and base matches the LoRA's training
distribution. Text encoding stayed on the v3-proven CLIPLoader/CLIPTextEncode
path; downloads ~15.6 GB in 140 s, session total 513.2 s, stopped clean.

Two risks that did not materialize: the hook node works fine on GGUF-loaded
models (the card admits its ComfyUI port is "vibe-coded", verified only
against diffusers — no `_fallback_merge` row was triggered), and VRAM held at
~13.3/15.64 GB with the 0.68 GB LoRA resident. Weight-provenance caveat for
future comparisons: unsloth's base Q4_K_M is 4.20 GB vs the dead abenzerps
file's 4.60 GB — different quant runs; the 12-step anchor row (69.1 s vs the
old 79.2/87.1 s walls) suggests it is also somewhat faster, so use in-session
anchors, not cross-repo JSONs, for ratios.

License note: Qwen Research, non-commercial — fine for eval/content probes,
not for a product.

**Shipped 2026-09-26:** this recipe is now the default path in the public
repo [run-qwen-image-on-free-colab](https://github.com/KodeIsFun/run-qwen-image-on-free-colab)
(commits 5d3aa28 + 33e9330): turbo notebook cells, `txt2img.py --turbo`
default + `--no-turbo` baseline, `workflows/t2i-turbo.json`, SKILL.md
invariants 10-12, and a `tools/check_workflows.py` consistency gate keeping
the three graph copies identical. Re-verified end-to-end on a fresh T4 (all
7 cells OK, 72.1 s cold generate matching this probe, tunnel-served proof
from outside). The update also fixes that repo's issue #1 — a user hit the
dead abenzerps GGUF URL on 2026-09-22 — which is what forced the unsloth
weight swap everywhere.

# W40 — The first free-T4 image-gen table: Z-Image-Turbo vs Qwen-Image turbo

Backlog item 3: Z-Image-Turbo (Tongyi-MAI, 6B S3-DiT, apache-2.0) claims
"fits comfortably within 16G VRAM" at 8 NFEs; nobody had published it next to
Qwen-Image-2.1-turbo on the same free card. Both stacks ran in single
sessions, seed 42, the two v4-v6 canary prompts, in-session rows for both
sides. Surface = Colab T4 via env proof (`COLAB_GPU=1`).

## The table (results/timings_v7_zimage.json + results/timings_v7b_warm.json)

| row (sign canary unless noted) | Z-Image bf16, 8-step | Qwen r256, 6-step |
|---|---|---|
| warm repeat, 768x768 (encode cached, v7b seed44) | **18.0 s** | 27.0 s |
| first render with a new prompt, 768x768 | 42.0 s | 30.0 s |
| warm, 1024x1024 | 39.0 s | 42.0 s |
| first image of a session (stack load included) | 102.1 s | 84.1 s |
| cat cold row (stack load included) | 102.1 s | 84.1 s |
| memory-friendly variant (z int8) | 60.0 s | — |

Per-step: Z bf16 1.78-1.91 s/step vs Qwen 3.8 s/step; z int8 4.14 s/step
(the convrot int8 path is slow on T4 despite fitting easiest).

## Findings

1. **The "fits 16GB" claim is true — but barely**: z bf16 unet 12.31 GB + TE
   qwen_3_4b bf16 8.04 GB cannot both stay resident on 15.6 GB, so ComfyUI
   unloads the TE between stages. Consequence: Z wins every *repeat* render
   (18.0 s = fastest measured on a free T4, 1.5x Qwen) but loses on
   *first* renders with fresh text (42.0 s — the TE reload) and on session
   start. Qwen's GGUF stack (4.2 GB unet + 9.35 GB TE both resident) is
   prompt-change-proof at 30.0 s.
2. **Workload rule**: batches of variations on one prompt → Z-Image; mixed
   prompts / interactive → Qwen.
3. **Z-Image recipe quirks** (official template, verbatim): CLIPLoader
   `type=lumina2` for the Qwen3-4B TE (there is no z_image type),
   `ModelSamplingAuraFlow shift=3`, KSampler 8 steps cfg 1
   res_multistep/simple, EmptySD3LatentImage. The `distill_patch_lora` in the
   Comfy-Org repo is not part of the t2i template.
4. **Quality**: "FREE GPU LAB" letter-perfect on Z bf16 at 768 and 1024, on z
   int8, and on Qwen — text rendering is a tie on this canary.
5. Cross-VM stability: v7 ran on VM1 (recycled before images were saved —
   numbers survived via the job log), then re-ran identically on VM2
   (results/): qwen rows match to the second (30.0/42.0), Z rows within ±10%
   (45.0→42.0, 33.0→39.0). results/ carries the VM2 rerun + v7b pair so every
   PNG and timing share one VM.

## Recipes side by side

- Z-Image: Comfy-Org/z_image_turbo split files, stock ComfyUI (nodes_zimage
  native), apache-2.0.
- Qwen: unsloth GGUF Q4_K_M + abenzerps TE/VAE + Viggle r256 runtime-hook
  LoRA + resolution-shifted 6-step sigmas, cfg off (v6 recipe verbatim).

## Attempt history

- v7 VM1: full matrix OK (1010.6 s), but the VM recycled before PNG download;
  timings JSON recovered from the job log, then superseded by the VM2 rerun.
- v7b attempt 1: assumed v7's cached weights, died on boot (fresh VM, no
  /content/ComfyUI) — fixed by giving v7b its own cached-skip setup.
- v7b attempt 2: NameError from an URL-constant ordering bug — fixed, then
  complete (743.7 s).
- v7 rerun on VM2 (472.7 s): the coherent image set in results/.

## Tweets

`tweets/01-free-t4-image-table.txt` — gate PASS against both timings files.
Unposted.

# 2026-W40 — BFS (Best Face Swap) on a free T4 — full test

Owner's prompt: "next experiment is the following, we need to do the full test:
<https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap>" → `PLAN.md`.

BFS is a popular LoRA collection (~150k downloads/mo, MIT) for face/head/body
replacement. We tested the **Qwen-Image-2.1 family** — the only one that rides
the lab's proven W39/W40 stack (7B DiT, qwen3vl int8 TE, bf16 VAE, fp16 cast +
`--disable-comfy-compiler`, RES4LYF `deis_2m`). Flux-2-Klein-4B and Krea-2
variants need base models the lab has never run — recorded as follow-ups.

## Verdict (short)

**BFS works end-to-end on a free T4, but not at the official template's own
settings, and its identity numbers come with a twist.**

- The template binds both reference images at the latent size (2 MP). Once the
  two-image conditioning actually works, that is a **~40k-token DiT sequence →
  258.8 s/step** on the T4 (vs 11 s/step with the images silently unbound).
  Pre-scaling references down (scene 768², head 512² — the BFS author's own
  "give the reference room, not resolution" rule from the body-swap doc) makes
  it tractable; the latent stays at template sizes.
- At **1024²** the head swap is real: the reference's red curly hair, freckles
  and green eyes land on the scene body, the night-street background and the
  **"FREE GPU LAB" neon canary survive letter-perfect**, and the image stays
  photographic. At the template's 1408² latent, identity transfer *fails*
  (scene hair kept) and the whole frame takes the posterized/HDR hit the
  workflow's own Note warns about — 1024² is the T4 sweet spot.
- **Identity twist (corrected 2026-09-30 on visual re-review):** with the
  two-image conditioning working, the no-BFS anchor *also* swaps the head —
  red curly hair included — and scores *higher* ArcFace similarity to the
  reference (0.867) than the BFS rows (0.640 @1024, 0.429 @1408). That
  strongly confirms the BFS guide's own admission that the base model
  "already approximates pasting the reference head". What BFS measurably adds
  on this pair is **scene coherence**: it is the only row that keeps the scene
  person's beret (the anchor pastes a bare head and drops the hat), at the
  cost of face likeness; and at the template's 1408² the BFS row loses
  identity outright (scene hair kept) while taking the posterized hit.
  (An earlier draft of this README wrongly wrote the anchor "keeps blonde
  hair" — R4's PNG shows otherwise.) Background fidelity costs nothing
  (SSIM vs scene 0.39–0.42 for every row incl. the anchor; unrelated-image
  floor 0.083).

## Headline numbers (all in `results/timings_bfs.json`)

| row | stack | wall | s/step | sim→refhead | sim→sceneperson | bg SSIM |
|---|---|---|---|---|---|---|
| R1 official @1408 cold | int8 + merge Pruna8+BFS v1.1 | 477.6 s | 51.96 | 0.4292 | 0.3131 | 0.3837 |
| R2 same, warm (crop trick) | " | 438.4 s | 47.92 | 0.4292 | 0.3131 | 0.3837 |
| **R3 head swap @1024** | int8 + merge Pruna8+BFS | **222.2 s** | 20.71 | **0.6399** | 0.2170 | 0.3906 |
| R4 anchor: no BFS @1024 | int8 + merge Pruna8 | 168.1 s | 20.26 | 0.8670 | 0.0856 | 0.3966 |
| R7 body swap @1408 | int8 + merge Pruna8+BFS body | 270.3 s | 26.63 | 0.3810 | 0.1119 | 0.4233 |
| **R6 head swap, GGUF Q4** | GGUF + merge Pruna8+BFS | **96.1 s** | **10.54** | 0.6318 | 0.1967 | 0.4127 |
| R5 GGUF + hook loader | — | FAILED | — | — | — | — |

- **GGUF Q4_K_M is 2.3× faster than the official int8_convrot DiT on the same
  swap task** (96.1 s vs 222.2 s wall; 10.54 vs 20.71 s/step) with identical
  eyeball quality and near-identical metrics — the lab's proven weights stay
  the default.
- BFS LoRA cost over the anchor at 1024²: +54 s wall (+32%), all in the merge +
  heavier steps; background SSIM unchanged (0.3906 vs 0.3966).
- Body swap (R7): clothing/proportions from the reference land on the scene
  pose; face small-in-frame likeness is low (0.381) exactly as the guide's own
  limitations section predicts.
- R1 ≡ R2 pixel-identical (same metrics; sha differs only in embedded workflow
  metadata) — the warm "crop trick" (center-crop no-op changes the node hash)
  defeats ComfyUI's per-node execution cache without changing the image.

## What broke (each attempt shipped a fix)

1. **min_bytes floor vs reality** — 0.5 GB floor for a 0.336 GB Pruna8 LoRA →
   the resume-loop correctly refused the "short" file 8× and exited. Floors now
   come from the HF API size, not guesses.
2. **`QwenImage21Cache` needs `device`/`dtype` in API format** (required
   widgets, `auto`/`default` — the author's own widget values); plus a **zombie
   ComfyUI from the crashed attempt held port 8188** — the new server died at
   bind while the zombie answered preflight. `boot_server()` now pkills first
   and the script atexit-kills its server.
3. **The big one: nested-dict autogrow silently binds ZERO images.** ComfyUI's
   `_resolve_dynamic_paths` pops *flat dotted keys* (`"images.image_1"`) from
   the prompt dict; a `{"images": {...}}` dict validates AND executes as pure
   text-to-image. Attempt 2 rendered 6 confident-looking swap rows that had
   never seen either input image (outputs: a studio man for head_swap, a forest
   child for body_swap — caught only by eyeballing PNGs + missing LoadImage
   nodes in the executed list). Flat keys first, always.
4. **The viggle runtime-hook loader (exact-update LoRA path from W40) is dead
   on current ComfyUI master** for this model: `QwenImage21Transformer2DModel
   has no attribute 'diffusion_model'` — wrapper API drift since the Sept-24
   master it was built on. Stock merge is the working path (and matches what
   the BFS author ships).
5. **258.8 s/step at template-default references** — see T4 adaptation above
   (recorded in `t4_adaptation` in the JSON; source: attempt-3 live progress
   poll before the kill).

## Consent + reproducibility

All three input images are generated by the lab's viggle-r256 turbo pipeline at
seed 42 (scene with the FREE GPU LAB canary, reference head, full-body
reference in the exact format the body doc trains on) — no real person, no
public figure, fully reproducible from `job_bfs.py` + seed. ArcFace
(`insightface` buffalo_l, CPU) and block-SSIM computed on the same VM from the
downloaded artifacts (`eval_final` in the JSON).

## Files

| file | what |
|---|---|
| `results/timings_bfs.json` | single source of truth: attempt-4 rows, preflight, downloads, eval, T4-adaptation measurements |
| `results/eval_final.json` | ArcFace + SSIM + dims + sha256 (attempt-4 files) |
| `results/final/out/` | all attempt-4 PNGs (`*_00002_/_00003_`) + comfyui logs |
| `results/out/`, `results/out_run1/` | attempt-2 evidence (the silently-unbound-conditioning rows) |
| `results/timings_bfs_attempt1.json` | attempt-1 evidence (min_bytes kill) |
| `job_bfs.py` | the whole driver: ladders, matrix, self-diagnosing preflight |

Attempt-2's PNGs (`results/out/bfsR1_head1408_00001_.png` etc.) are kept as the
cautionary exhibit: confident-looking outputs from a graph that never saw the
inputs.

## Follow-ups

- FLUX.2-klein-4B BFS variant (smallest base; TE stack unmeasured on T4).
- Re-test the hook loader if upstream fixes the wrapper API.
- 2048² output (the workflow Note claims it mitigates the whole-image effect) —
  needs the ref-prefix KV cache to actually engage; at 52 s/step it is a
  40-minute image on a T4.

---

**Shipped 2026-10-01:** this recipe is now a public, independently verified
repo — [run-bfs-face-swap-on-free-colab](https://github.com/KodeIsFun/run-bfs-face-swap-on-free-colab)
(notebook, Kaggle kernel [emdadh/run-bfs-faceswap-t4](https://www.kaggle.com/code/emdadh/run-bfs-faceswap-t4),
stdlib client `clients/headswap.py`, workflow JSONs, consistency gate). The
actual notebook was executed headlessly on a fresh Colab T4 from a GitHub
clone (NBEXIT 0; fresh head swap pixel-identical to this project's GGUF row)
and ran COMPLETE as a public Kaggle kernel (head 132 s, body 90 s). Repo
commit 19ef486.

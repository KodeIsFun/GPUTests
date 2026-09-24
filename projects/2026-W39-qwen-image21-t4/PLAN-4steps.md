# Plan: 4-step probe — same v2b settings, no Pocket rewriter

**Status: RUN 2026-09-24 — quality FAIL, 12 steps stays default.** Results in `README.md` § "4-step follow-up" and `results-v3/`. Baseline was switched from v2b (non-UC weights) to the Pocket probe's direct arm (UC-Q4_K_M) at run time: the non-UC diffusion GGUF 404s after the abenzerps repo restructure. cfg 2.5 sign illegible at 4 steps; cfg 1.0 contingency (front-loaded into the same session) only partially legible. Speed held: 27.0 s warm vs 87.1 s (3.2×).

## Question

Qwen-Image-2.1 GGUF Q4_K_M on the free Colab T4 currently warms at ~79 s with 12 steps. Does the same
pipeline at **4 steps** hold quality while dropping warm renders to ~30 s? If yes, 4 steps becomes the
default for every future probe (≈2.6× faster iteration).

Scope guard: this is the **v2b lineage only** — same weights, same flags, direct prompts. The Pocket
rewriter arm from `2026-W39-qwen-image21-pocket-t4/` is out of scope. Note the Pocket probe used the
*Uncensored* GGUF (`qwen-image-2.1-UC-Q4_K_M`); this probe reuses v2b's `qwen-image-2.1-Q4_K_M` so the
12-step refs in `results/` stay a true apples-to-apples baseline. Do not mix the two GGUF files when
comparing.

## Fixed settings (identical to v2b — the only change is steps)

| Setting | Value |
|---|---|
| Diffusion weights | `qwen-image-2.1-Q4_K_M.gguf` (4.60 GB, abenzerps) |
| Text encoder / VAE | `qwen3vl_8b_int8_convrot.safetensors` / `qwen_image_2.1_vae_bf16.safetensors` |
| Flags | `--force-fp16 --disable-comfy-compiler` (compiler crash on T4 without it) |
| Sampler / scheduler | `res_multistep` / `simple`, denoise 1.0 |
| cfg / seed / size | 2.5 / 42 / 768×768 |
| **Steps** | **4** (baseline was 12) |

## Run

1. Derive `job_v3_steps4.py` from `job_v2b.py`: `STEPS = 12 → 4`, run tag
   `qwen-image-2.1-q4km-t4-v3-fp16-nocompile-steps4`. Nothing else changes — same download caching,
   same `/object_info` node pre-check, same timings.json schema.
2. Same two prompts as v2b, same seed: `p1_cat_raincoat`, `p2_freegpulab_sign` (the sign is the
   built-in text-rendering canary).
3. Colab lane exactly per repo rules: `colab new -s w39-4steps --gpu T4` → `colab upload` → detached
   launch → poll log → `colab download` artifacts → **`colab stop`**. No `--keep`, no
   auth/drivemount/repl. Colab free tier only — no Kaggle hours, no Proxy spend.
4. Post-run: build a side-by-side grid per prompt (4-step vs the v2b 12-step ref, same seed) into
   `results/`, and append a Results section to this project's README.

## Expected numbers (hypothesis to verify, not assume)

- Warm render ≈ 4 × 6.20 s/step + ~5 s fixed overhead ≈ **~30 s** (vs 79.17 s at 12 steps).
- s/step should stay ≈ 6.2 — cfg 2.5 still runs cond+uncond each step, so per-step cost is unchanged;
  the win is purely fewer steps.
- Total session ≈ 10–12 min including clone, pip, and ~185 s of weight downloads.

## Verdict criteria

- **Pass** → adopt 4 steps as the lab default for Qwen-Image-2.1 T4 probes. Sign still spells
  "FREE GPU LAB" correctly, cat keeps the yellow raincoat, no under-denoised smearing at 100% zoom.
- **Fail on quality** → record it, keep 12 steps as default, and run **one** contingency warm render at
  cfg 1.0 (CFG off) for the sign prompt — few-step sampling sometimes needs CFG off. If that also
  fails, 4 steps is closed; no further ladder.
- **Fail on timing** (s/step drifts well above 6.2) → investigate before re-running; don't tune blind.

## Evidence rule

Any number in a tweet about this run must appear in the new timings JSON. Spelled-out numbers don't
pass the gate — digits or nothing.

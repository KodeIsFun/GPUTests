# W40 — Ternary-Bonsai-2-27B on a Kaggle T4 (the CUDA truth for a Mac-famous model)

Backlog item 4, reviving the GGUF-on-T4 series (W38, the lab's
best-performing content). prism-ml's Ternary-Bonsai-2-27B GGUF repo has 3.46M
downloads and every published benchmark is Apple Silicon (card: ~47 tok/s on
an M5 Max). Nobody had published it on the free NVIDIA card.

## Scouting findings (before any GPU time)

- **Stock llama.cpp cannot load these GGUFs**: PQ2_0/PTQ1_0 are fork-specific
  GGML types (142/143; upstream feature request from 2026-09-18 still open),
  and the demo repo explicitly warns a Q2_0 path "can load without the
  required transforms and produce gibberish". The PrismML fork with the
  custom ternary hybrid-attention CUDA/Metal kernels is mandatory.
- **No toolchain problem anyway**: the fork pins prebuilt binaries at release
  `prism-b10743-adfffbe` (incl. `linux-cuda-12.4`), sidestepping the W38
  "no usable nvcc on Kaggle" lesson.
- Comparison quant chosen for footprint: unsloth `Qwen3.8-27B-UD-IQ2_XXS`
  (7.27 GB ≈ PQ2_0's 7.21 GB). The card's own quality claim: ternary retains
  98.2 % of FP16 intelligence where the conventional IQ2 build collapses
  (84.78 vs 72.59 on their 14-bench suite) — we measure speed, not quality.

## Results (Kaggle T4, driver 580.159.04, fork build b10743, -ngl 99, pp512/tg128 ×3 reps)

| model | file | pp512 tok/s | tg tok/s |
|---|---|---|---|
| Ternary-Bonsai-2-27B, PQ2_0 packing | 7.21 GB | 294.66 | **21.56** |
| Ternary-Bonsai-2-27B, PTQ1_0 packing | 5.95 GB | 294.38 | 17.77 |
| Qwen3.8-27B, conventional UD-IQ2_XXS | 7.27 GB | 246.29 | 11.86 |

**The headline: at the same footprint, the ternary kernels generate 1.8×
faster than the conventional quant on a free T4** (21.56 vs 11.86 tok/s), and
prompt processing is also faster (294.66 vs 246.29). The denser PTQ1_0
packing trades 1.26 GB for ~20 % slower decode than PQ2_0 — packing density
is not monotonic with speed here. Cross-version stability: kernel v1 (parser
bug, numbers unreferenced) measured tg 17.90/22.07/11.51 — matches v2 within
noise. Every smoke generation (200 tokens, greedy) completed exit-0.

VRAM is **not instrumented** in this environment: the container's nvidia-smi
`memory.used` stays at ~0.1-4 GB while 5.5-7.3 GB models run at ngl 99 on
CUDA — it does not reflect process allocations (see per-row `vram_note`).

The model reasons before answering (card's known-issues: give `-n 16384`,
`-c 65536`; `reasoning_effort: medium` for shorter outputs) — irrelevant to
raw tok/s, which is what this table measures.

## Run history

- **v1** (30.4 min): benches fine, but the parser expected the classic wide
  llama-bench table; this fork emits the long format — numbers unparsed, so
  the smoke tests (gated on parsed tps) never ran. `success: false` by
  design.
- **v2** (13.2 min): long-format parser, smoke ungated, VRAM trace (which
  exposed the nvidia-smi limitation above). `success: true`. THE artifact.
- v1→v2 push churn taught a Kaggle CLI wart: when the metadata `id` and the
  title-derived slug diverge, re-pushes 409 (`emdadh/ternary-bonsai-t4` vs
  the real `emdadh/ternary-bonsai-2-27b-on-t4`). Fix: set `id` to the
  title-derived slug.

## Files

- `job.py` + `kernel-v8/` — the kernel (v2 = what ran; public at
  https://www.kaggle.com/code/emdadh/ternary-bonsai-2-27b-on-t4)
- `results/results_v8_ternary.json` — THE numbers (schema uses the lab's
  whitelisted evidence keys: `pp512_tok_s`, `tg_tok_s`, `file_gb`, plus
  `models: 2`)
- `results/kernel-v2.log` — full kernel log
- `tweets/01-ternary-on-t4.txt` — gate PASS against the results JSON

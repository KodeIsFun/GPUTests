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

---

## Local lane (2026-10-12): the SAME ternary model runs on the owner's AMD RX 5500 XT — faster than the T4 at generation

The owner's desktop (Ryzen 5 5600, 31 GB RAM, **AMD Radeon RX 5500 XT 8 GB**, no NVIDIA)
can run this exact workload 100% on GPU — the PrismML fork ships **Vulkan kernels**
for the ternary types (`dequant_pq2_0.comp`, `mul_mat_vecq_ptq1_0.comp` in
`ggml-vulkan/vulkan-shaders/`), so no CUDA is needed. Verified end-to-end with the
fork's **prebuilt `ubuntu-vulkan-x64` binaries** (pinned release `prism-b10743-adfffbe`,
no compiler, no sudo) against `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.95 GB, HEAD-verified)
placed in the ollama models dir on the data drive.

| test | local RX 5500 XT (Vulkan, ngl 99) | free Kaggle T4 (CUDA) |
|---|---|---|
| tg128 | **18.57 ± 0.03 tok/s** | 17.77 tok/s |
| pp512 | 39.25 ± 0.13 tok/s | ~330 tok/s (CUDA tensor cores) |

Generation is **slightly faster than the free T4** while the CPU (12 threads) and
31 GB RAM stay completely free; prefill is ~8x slower (RADV NAVI14 reports
"matrix cores: none" — T4 tensor cores win prompt processing by a mile). The
PQ2_0 headline quant (7.21 GB) does not fit in 8 GB VRAM and stays a T4 lane.

**The setup recipe (all user-space, reproducible)** — three mismatches were paid for:

1. The box has no CUDA; Ollama's GPU use is its own bundled ROCm (irrelevant here).
   The ternary fork does not support ROCm on RDNA1 — **Vulkan is the only GPU lane**
   for this card, via Mesa RADV (already installed, `mesa-vulkan-drivers` 26.0.4).
2. The Mesa PPA (`ernstp/mesarc`) half-upgraded the stack: RADV 26 needs
   `wl_fixes_interface` from libwayland 1.25 (installed via
   `sudo apt install -y libwayland-client0`), but an old `/opt/amdgpu/lib` copy
   shadowed it — fixed with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` prefix.
3. The system Vulkan loader (jammy, 1.3.204/2022) rejects the new RADV
   (`VK_EXT_layer_settings` unknown), and Ubuntu 24.04's loader deb needs a newer
   glibc than jammy has — the working loader is LunarG's **jammy-built**
   `libvulkan1_1.4.313.0~rc1-1lunarg22.04` deb, extracted with `dpkg -x` (no install).

Everything lives in `~/ternary-local/`: `run-vulkan.sh` (wrapper exporting the
LD_LIBRARY_PATH and pinning `-ngl 99`), `llama-prism-b10743-adfffbe/` (prebuilt),
`vulkan-usr/` (the extracted loader). Smoke generation: the reasoning trace
(`[Start thinking] ... Final: Paris.`) renders on the GPU with exit-0 semantics.

### OpenAI-compatible server on the same GPU (2026-10-12, later)

The same prebuilt ships `llama-server`, so the local lane is now an
**OpenAI-API server** (`~/ternary-local/serve-bonsai.sh start|stop`,
`http://127.0.0.1:8080/v1`, model id `ternary-bonsai-2-27b`, ctx 8192,
`-fa auto`, alias set). Verified: `GET /v1/models`; non-streaming
`POST /v1/chat/completions` (clean `choices[0].message.content`,
`usage` block, `finish_reason: stop`); streaming SSE chunks with
`reasoning_content` deltas carrying the `[Start thinking]` trace
separately from the answer. Footprint while serving: **7.65 GB VRAM,
system RAM at desktop baseline (~5 GB), CPU idle** — fully offloaded
(-ngl 99), flash-attn auto. Point any OpenAI-format client at:

    base_url = http://127.0.0.1:8080/v1    api_key = any-string    model = ternary-bonsai-2-27b

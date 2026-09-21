# 06 — Troubleshooting: every gotcha, with its fix

All of these were hit for real on free-tier VMs. Symptom → cause → fix.

## §1 Source build of llama-cpp-python dies instantly

**Symptom:** `pip install llama-cpp-python` (from PyPI sdist) fails in <60 s;
the error tail is pip boilerplate ("Building wheel ... did not run
successfully") with no usable compiler message.
**Cause:** something in the image's toolchain; nvcc 12.8 is present, so it is
NOT "no CUDA". Not worth diagnosing.
**Fix:** the prebuilt wheel:
`pip install "llama-cpp-python[server]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`
(~66 s). If you must debug a build, capture the FULL log to a file first —
a truncated tail is how the cause got lost the first time.

## §2 "GPU" runtime that isn't

**Symptom:** model runs but at 2–4 tok/s; no `offloaded` lines in logs.
**Fix:** verify the runtime, not the flag: `nvidia-smi` must show a Tesla T4;
after load, llama.cpp must print `offloaded N/N layers to GPU` (capture init
stderr). If llama.cpp reports GPU offload unsupported, STOP — do not publish
CPU numbers as GPU numbers.

## §3 VRAM precheck skips a model that would have run

**Symptom:** "skipped_vram" for a ~12 GB model on a 15 GB card.
**Cause:** over-fat safety margins (e.g. 3.5 GB). MoE KV at 4k ctx is ~200 MB.
**Fix:** honest reserve (`file_GB×1024 + 1500 MB`), attempt the load, and
treat an actual OOM as a *result* (record it, continue) — not as a reason to
skip harder next time.

## §4 OOM on load of the big model

**Symptom:** `RuntimeError` / llama.cpp allocation failure during load.
**Fix:** in order — (1) smaller quant (Q4_K_S/Q3_K_M), (2) `n_ctx 2048`,
(3) smaller model. Record which combination ran; that's the recipe the user
gets.

## §5 Impossible benchmark numbers

**Symptom:** prompt-processing tok/s that beats the card's bandwidth (e.g.
>20k tok/s on 8B), or decode speed that exceeds `320 ÷ file_GB × 2`.
**Cause:** benchmarking on a dirty context (chat history still resident) or
CUDA graph capture amortizing into a warm pass.
**Fix:** prompt-processing first on a clean context: eval once untimed →
`reset()` → eval timed → `reset()`. Never measure after a chat turn.

## §6 `colab run` dies in ~30 s

**Cause:** `--timeout` defaults to 30 seconds.
**Fix:** always pass `--timeout 20` (minutes) or larger.

## §7 colab CLI crashes with `AttributeError: ... KernelClient`

**Cause:** a dependency upgrade (`jupyter_kernel_client` 1.0.2 renamed it).
**Fix:** `uv pip install --python ~/.local/share/uv/tools/google-colab-cli/bin/python 'jupyter_kernel_client<1'`
Then `colab sessions` to confirm.

## §8 gpt-oss-20b answers wrapped in `<|channel|>analysis<|message|>...`

**Cause:** the harmony chat template does not auto-apply through
llama-cpp-python's `create_chat_completion`/server.
**Impact:** decode speed is unaffected and the model still reasons coherently
— but the text needs parsing, and other clients will see markers.
**Fix:** prefer Qwen3 for API serving; if you must serve gpt-oss, strip
`<|channel|>...<|message|>` prefixes client-side, or post-process:
split on `<|message|>` and keep the `final` channel when present.

## §9 Qwen3 burns the whole token budget thinking

**Symptom:** `max_tokens=640` of output and the answer field is empty (it's
all inside `<think>...</think>`).
**Fix:** pass template kwargs where supported:
`create_chat_completion(..., chat_template_kwargs={"enable_thinking": False})`,
falling back to plain call on TypeError. For measurement, note that thinking
does not change tok/s — only whether you get a usable answer in-budget.

## §10 `kernels output` silently returns stale files

**Cause:** the CLI skips download when a local file is newer.
**Fix:** `kaggle kernels output <user>/<slug> -p out/ --force` whenever
re-harvesting a newer version.

## §11 Kaggle push 409 on every retry

**Cause:** Kaggle derives the kernel slug from the TITLE; a metadata id that
disagrees creates a kernel under the title slug, and later pushes of the id
conflict.
**Fix:** make `slugify(title) == id-slug` exactly. Never "fix" it by renaming
the existing kernel — the URLs are already live.

## §12 `kaggle b`-style 401 / expired token (only if using Benchmarks)

**Fix:** re-run `kaggle b init -y` (restores `LLM_DEFAULT`) then
`kaggle b auth -y` (1-hour key). And never `exec()` a task file with fresh
auth — module-level `.evaluate()` spends real money immediately.

## §13 Server won't bind on a Colab VM

**Symptom:** `[Errno 98] error while attempting to bind on address
('0.0.0.0', 8080): address already in use` in the server log.
**Cause:** Colab's own services already occupy 8080 on the 2026 image.
**Fix:** never hardcode the port — bind a socket to port 0, take the OS-assigned
port, use it for the server and the tunnel (the notebook does this). Also
poll `server.poll()` while waiting for `/health` so a crashed server fails
fast instead of burning the whole health-wait loop.

## §14 `/health` returns 404 on the llama-cpp-python server

**Symptom:** server is up (uvicorn logs requests) but every `GET /health`
says 404, so a health-wait loop never exits.
**Cause:** 0.3.x of llama-cpp-python does not expose `/health`.
**Fix:** poll `GET /v1/models` — the app only accepts requests after the
model has loaded, so the first 200 is the readiness signal.

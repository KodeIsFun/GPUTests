"""W5 "GGUF on T4": what actually runs on the free Kaggle GPU, and how fast.

The Month-2 pivot question: everyone with a free Colab/Kaggle GPU asks
"can I run a 7B/14B/open-weights model locally, and at what speed?" Nobody
publishes a measured table, so this week produces one.

One deliberately self-contained file (no lab/ imports), pushed to a Kaggle T4
with a 2 h platform budget (-t 7200). Plan-week W5, month 2.

Measures, per model, smallest -> largest so the big ones inherit a warm build
and an abort leaves the small results already written:
  1. GGUF download from Hugging Face (timed, file size recorded)
  2. llama.cpp load with all layers on the GPU (timed) + VRAM after load
  3. warmup chat turn (8 tokens, greedy)
  4. generation throughput: one greedy chat turn, max_tokens=640, tiny prompt
     -> tg_tok_s = completion_tokens / wall
  5. prompt processing: llm.eval on exactly 512 tokens -> pp512_tok_s
  6. VRAM peak observed during the run
  7. sanity: did the answer actually answer (content after any <think> block
     starts with a letter/digit)? gibberish or empty means the offload lied.

Model set (all ungated on HF, verified 2026-09-20):
  bartowski/Qwen_Qwen3-4B-GGUF    Qwen_Qwen3-4B-Q4_K_M.gguf
  bartowski/Qwen_Qwen3-8B-GGUF    Qwen_Qwen3-8B-Q4_K_M.gguf
  bartowski/Qwen_Qwen3-14B-GGUF   Qwen_Qwen3-14B-Q4_K_M.gguf
  ggml-org/gpt-oss-20b-GGUF       gpt-oss-20b-MXFP4.gguf

An OOM is a result, not a failure: the manifest says "14B runs or we publish
the OOM", and the same is true for gpt-oss-20b.

Writes timings.json incrementally after every stage, and echoes it between
###TIMINGS### markers at the end.
"""
from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Stamped at push time with the commit this job was built from.
GIT_SHA: str | None = "98fbe87"

BUDGET_S = 7200
N_CTX = 4096
TG_MAX_TOKENS = 640
PP_TOKENS = 512
QUESTION = "Name three planets in the solar system and one fact about each."
VRAM_SAFETY_MARGIN_MB = 2500  # weights + this must fit in free VRAM to attempt a load
KV_RESERVE_MB = 1000  # rough KV-cache + CUDA context headroom

MODELS: list[dict] = [
    {
        "model": "qwen3-4b-q4-k-m",
        "repo_id": "bartowski/Qwen_Qwen3-4B-GGUF",
        "filename": "Qwen_Qwen3-4B-Q4_K_M.gguf",
        "chat_kwargs": {"enable_thinking": False},
    },
    {
        "model": "qwen3-8b-q4-k-m",
        "repo_id": "bartowski/Qwen_Qwen3-8B-GGUF",
        "filename": "Qwen_Qwen3-8B-Q4_K_M.gguf",
        "chat_kwargs": {"enable_thinking": False},
    },
    {
        "model": "qwen3-14b-q4-k-m",
        "repo_id": "bartowski/Qwen_Qwen3-14B-GGUF",
        "filename": "Qwen_Qwen3-14B-Q4_K_M.gguf",
        "chat_kwargs": {"enable_thinking": False},
    },
    {
        "model": "gpt-oss-20b-mxfp4",
        "repo_id": "ggml-org/gpt-oss-20b-GGUF",
        "filename": "gpt-oss-20b-MXFP4.gguf",
        "chat_kwargs": {"reasoning_effort": "low"},
    },
]

FILLER = (
    "The quick brown fox jumps over the lazy dog while the moon rises over "
    "the quiet harbor and the streetlights hum their low electric song. "
)

GPU_QUERY = "name,memory.total,memory.used,driver_version"


def output_dir() -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.exists():
        return kaggle
    fallback = Path("results")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def run(cmd: list[str], timeout: int = 2700, env: dict | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def query_gpu() -> list[dict]:
    code, raw = run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        timeout=120,
    )
    rows = []
    if code == 0:
        for line in raw.strip().splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) >= 4:
                rows.append(
                    {
                        "name": fields[0],
                        "vram_total_mb": float(fields[1]),
                        "vram_used_mb": float(fields[2]),
                        "driver": fields[3],
                    }
                )
    return rows


RESULTS_PATH: Path | None = None
RESULTS: dict = {}


def flush() -> None:
    if RESULTS_PATH is not None:
        RESULTS_PATH.write_text(json.dumps(RESULTS, indent=2) + "\n")


def build_llama_cpp() -> float:
    t0 = time.time()
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    env["CMAKE_CUDA_ARCHITECTURES"] = "75"  # T4 is sm_75; one arch keeps the build short
    env["FORCE_CMAKE"] = "1"
    code, out = run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "llama-cpp-python"],
        timeout=2700,
        env=env,
    )
    if code != 0:
        RESULTS["build_error"] = out[-4000:]
        flush()
        raise SystemExit("llama-cpp-python CUDA build failed; see build_error")
    return time.time() - t0


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return (text or "").strip()


def answer_ok(text: str) -> bool:
    body = strip_thinking(text)
    return bool(body) and bool(re.match(r"[A-Za-z0-9]", body))


def download(entry: dict) -> tuple[str, float, float]:
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    path = hf_hub_download(repo_id=entry["repo_id"], filename=entry["filename"])
    size_gb = Path(path).stat().st_size / 1e9
    return path, time.time() - t0, size_gb


def load_model(path: str):
    from llama_cpp import Llama

    return Llama(
        model_path=path,
        n_gpu_layers=-1,
        n_ctx=N_CTX,
        verbose=False,
    )


def chat(llm, entry: dict, messages: list, max_tokens: int):
    """Chat turn; retries without template kwargs on engines that lack them."""
    try:
        return llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            chat_template_kwargs=entry["chat_kwargs"],
        )
    except (TypeError, ValueError, KeyError):
        return llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.0
        )


def measure(entry: dict, baseline_used_mb: float) -> dict:
    rec: dict = {
        "model": entry["model"],
        "repo": entry["repo_id"],
        "file": entry["filename"],
        "n_ctx": N_CTX,
        "status": "ok",
    }

    gpu_rows = query_gpu()
    free_mb = (
        gpu_rows[0]["vram_total_mb"] - gpu_rows[0]["vram_used_mb"] if gpu_rows else 0.0
    )

    path, download_s, size_gb = download(entry)
    rec["file_gb"] = round(size_gb, 3)
    rec["download_s"] = round(download_s, 2)

    if gpu_rows and size_gb * 1024 + KV_RESERVE_MB + VRAM_SAFETY_MARGIN_MB > free_mb:
        rec["status"] = "skipped_vram"
        rec["free_mb_before_load"] = round(free_mb, 1)
        return rec

    t0 = time.time()
    llm = load_model(path)
    rec["load_s"] = round(time.time() - t0, 2)

    rows = query_gpu()
    rec["vram_after_load_mb"] = rows[0]["vram_used_mb"] if rows else None
    peak = rec["vram_after_load_mb"] or 0.0

    chat(llm, entry, [{"role": "user", "content": "Say OK."}], 8)

    # Generation throughput: tiny prompt, greedy, so wall time is ~pure decode.
    t0 = time.time()
    out = chat(
        llm,
        entry,
        [{"role": "user", "content": QUESTION}],
        TG_MAX_TOKENS,
    )
    tg_wall = time.time() - t0
    peak = max(peak, (query_gpu() or [{"vram_used_mb": 0}])[0]["vram_used_mb"])
    usage = out.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    content = ""
    if out.get("choices"):
        content = (out["choices"][0].get("message") or {}).get("content") or ""
        if not content:
            content = out["choices"][0].get("text") or ""
    rec["tg_wall_s"] = round(tg_wall, 2)
    rec["tg_completion_tokens"] = completion_tokens
    if completion_tokens and tg_wall > 0:
        rec["tg_tok_s"] = round(completion_tokens / tg_wall, 2)
    rec["answer_head"] = strip_thinking(content)[:160]
    rec["answer_ok"] = answer_ok(content)

    # Prompt processing: exactly PP_TOKENS tokens through llm.eval.
    tokens: list[int] = []
    while len(tokens) < PP_TOKENS:
        tokens.extend(llm.tokenize(FILLER.encode("utf-8")))
    llm.eval(tokens[:16])  # warm the eval path (CUDA graph capture etc.)
    t0 = time.time()
    llm.eval(tokens[:PP_TOKENS])
    pp_wall = time.time() - t0
    rec["pp512_tok_s"] = round(PP_TOKENS / pp_wall, 2) if pp_wall > 0 else None
    rec["vram_peak_mb"] = peak

    del llm
    gc.collect()
    freed = False
    for _ in range(15):
        rows = query_gpu()
        if rows and rows[0]["vram_used_mb"] <= baseline_used_mb + 250:
            freed = True
            break
        time.sleep(2)
    rec["vram_freed_after_del"] = freed
    return rec


def main() -> None:
    global RESULTS_PATH, RESULTS
    RESULTS_PATH = output_dir() / "timings.json"
    started = time.time()

    rows = query_gpu()
    RESULTS = {
        "metric": "gguf-llm-toks",
        "git_sha": GIT_SHA,
        "platform": "kaggle" if Path("/kaggle/working").exists() else "local",
        "gpu": rows[0]["name"] if rows else None,
        "vram_mb": rows[0]["vram_total_mb"] if rows else None,
        "driver": rows[0]["driver"] if rows else None,
        "baseline_vram_used_mb": rows[0]["vram_used_mb"] if rows else None,
        "budget_s": BUDGET_S,
        "models": [],
    }
    flush()

    RESULTS["build_s"] = round(build_llama_cpp(), 1)
    try:
        RESULTS["llama_cpp_python_version"] = importlib.metadata.version(
            "llama-cpp-python"
        )
    except importlib.metadata.PackageNotFoundError:
        RESULTS["llama_cpp_python_version"] = None
    flush()

    for entry in MODELS:
        if time.time() - started > BUDGET_S * 0.85:
            RESULTS.setdefault("skipped_time_budget", []).append(entry["model"])
            flush()
            break
        baseline = (query_gpu() or [{"vram_used_mb": 0}])[0]["vram_used_mb"]
        try:
            RESULTS["models"].append(measure(entry, baseline))
        except SystemExit:
            raise
        except BaseException as exc:  # noqa: BLE001 - an OOM is a result, keep going
            RESULTS["models"].append(
                {
                    "model": entry["model"],
                    "repo": entry["repo_id"],
                    "file": entry["filename"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
        flush()

    RESULTS["wall_s"] = round(time.time() - started, 1)
    flush()
    payload = json.dumps(RESULTS, indent=2)
    print("###TIMINGS###")
    print(payload)
    print("###TIMINGS###")


if __name__ == "__main__":
    main()

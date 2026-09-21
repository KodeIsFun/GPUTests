"""Unattended LLM benchmark for a free Kaggle T4 — self-contained (no repo deps).

This is the job that produced the measured table (Qwen3-4B/8B/14B Q4_K_M +
gpt-oss-20b MXFP4, llama.cpp on the T4): decode tok/s, prompt-512 tok/s on a
clean context, load/download times, VRAM and offload proof per model.

Protocol notes that are easy to get wrong (each one cost a kernel run once):
  - llama-cpp-python comes from the PREBUILT cu124 wheel index; the
    from-source build dies on these images with an unreadable error.
  - llama.cpp's "offloaded N/N layers" line is captured as proof the weights
    really reached the GPU (nvidia-smi under-reports on these drivers).
  - prompt processing runs FIRST on a clean context: untimed pass pays CUDA
    graph capture, reset(), then the timed pass. Measuring after a chat turn
    produces numbers that violate physics.
  - VRAM precheck uses an honest 1500 MB reserve: MoE KV at 4k ctx is ~200 MB,
    and over-fat margins made v2 skip gpt-oss-20b without attempting it. An
    OOM is caught and recorded as a RESULT, never a skip.
  - results are flushed to /kaggle/working/timings.json after EVERY stage, so
    a timeout still leaves the partial table harvestable.

Usage: put this file + a kernel-metadata.json (see references/03-kaggle-lane.md)
in a folder, edit MODELS/GIT_SHA below, then:
    kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4 -t 7200
    kaggle kernels status <user>/<slug>      # poll
    kaggle kernels output <user>/<slug> -p out/ --force
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Provenance stamp: set this to your commit/URL before pushing.
GIT_SHA: str | None = "SET-ME"
# Stamped at push time with the commit this job was built from.

BUDGET_S = 7200
N_CTX = 4096
TG_MAX_TOKENS = 640
PP_TOKENS = 512
QUESTION = "Name three planets in the solar system and one fact about each."
# v3: v2's fat margins (3500 MB) skipped gpt-oss-20b without even trying, and
# MoE KV at 4k ctx is ~200 MB. Honest budget: weights + 1500 MB for context,
# CUDA context and buffers — then an OOM is caught and recorded as a result.
VRAM_ATTEMPT_RESERVE_MB = 1500

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


WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"


def _nvcc_probe() -> dict:
    """Where is a CUDA compiler, if anywhere? Recorded either way so a build
    failure is diagnosable from the harvested json alone."""
    probe = {"nvcc_on_path": None, "nvcc_in_usr_local_cuda": None, "nvcc_version": None}
    code, out = run(["which", "nvcc"], timeout=30)
    probe["nvcc_on_path"] = code == 0 and bool(out.strip())
    probe["nvcc_in_usr_local_cuda"] = Path("/usr/local/cuda/bin/nvcc").exists()
    for cand in ("nvcc", "/usr/local/cuda/bin/nvcc"):
        code, out = run([cand, "--version"], timeout=60)
        if code == 0:
            match = re.search(r"release (\d+\.\d+)", out)
            probe["nvcc_version"] = match.group(1) if match else "unknown"
            break
    return probe


def build_llama_cpp() -> float:
    t0 = time.time()
    RESULTS["nvcc_probe"] = _nvcc_probe()

    # Path 1 (W5 v2): prebuilt CUDA wheel from the llama-cpp-python index.
    # The v1 source build died in 28 s — no usable nvcc in the runtime image —
    # and the index ships a py3-none manylinux wheel of the same version.
    code, out = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "llama-cpp-python",
            "--extra-index-url",
            WHEEL_INDEX,
        ],
        timeout=1200,
    )
    RESULTS["install_path"] = "prebuilt-cu124-wheel" if code == 0 else "wheel-failed"

    # Path 2: source build, only if the wheel is unusable. Full log captured —
    # v1 kept only the tail and the real compiler error was lost with it.
    if code != 0:
        RESULTS["wheel_install_tail"] = out[-2000:]
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        env["CMAKE_CUDA_ARCHITECTURES"] = "75"  # T4 is sm_75
        env["FORCE_CMAKE"] = "1"
        code, out = run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "llama-cpp-python",
            ],
            timeout=2700,
            env=env,
        )
        RESULTS["install_path"] += "+source-build"
        try:
            (output_dir() / "build.log").write_text(out)
        except OSError:
            pass

    if code != 0:
        RESULTS["build_error"] = out[-4000:]
        flush()
        raise SystemExit("llama-cpp-python install failed on every path; see json")

    import llama_cpp

    RESULTS["llama_cpp_python_version"] = llama_cpp.__version__
    supports = getattr(llama_cpp.llama_cpp, "llama_supports_gpu_offload", None)
    RESULTS["gpu_offload_supported"] = bool(supports()) if supports else None
    flush()
    # A CPU-only measurement must never ship dressed up as a T4 number.
    if RESULTS["gpu_offload_supported"] is False:
        raise SystemExit("llama.cpp reports NO GPU offload; refusing to measure")
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


def load_model(path: str) -> tuple[object, str]:
    """Load with llama.cpp's own init log captured — the offload line is the
    evidence that the weights actually went to the GPU (v2's nvidia-smi
    reading right after load under-reported vs the file size)."""
    import io
    from contextlib import redirect_stderr

    from llama_cpp import Llama

    captured = io.StringIO()
    with redirect_stderr(captured):
        llm = Llama(
            model_path=path,
            n_gpu_layers=-1,
            n_ctx=N_CTX,
            verbose=True,
        )
    return llm, captured.getvalue()


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
    rec["free_mb_before_load"] = round(free_mb, 1)

    path, download_s, size_gb = download(entry)
    rec["file_gb"] = round(size_gb, 3)
    rec["download_s"] = round(download_s, 2)

    if gpu_rows and size_gb * 1024 + VRAM_ATTEMPT_RESERVE_MB > free_mb:
        rec["status"] = "skipped_vram"
        return rec

    t0 = time.time()
    llm, init_log = load_model(path)
    rec["load_s"] = round(time.time() - t0, 2)
    offload_lines = [
        line.strip()
        for line in init_log.splitlines()
        if ("offloaded" in line.lower())
        or ("cuda" in line.lower() and "error" in line.lower())
    ]
    if offload_lines:
        rec["offload_log"] = offload_lines[:4]

    # Prompt processing on a CLEAN context, twice: v2 measured eval() on a
    # context already full of chat history and got pp numbers that violated
    # physics (27890 tok/s for 8B). First pass pays one-time CUDA graph
    # capture; the second pass on a reset context is the honest number.
    tokens: list[int] = []
    while len(tokens) < PP_TOKENS:
        tokens.extend(llm.tokenize(FILLER.encode("utf-8")))
    tokens = tokens[:PP_TOKENS]
    llm.eval(tokens)
    llm.reset()
    t0 = time.time()
    llm.eval(tokens)
    pp_wall = time.time() - t0
    llm.reset()
    rec["pp512_tok_s"] = round(PP_TOKENS / pp_wall, 2) if pp_wall > 0 else None

    # Warmup chat turn, then VRAM with weights fully resident.
    chat(llm, entry, [{"role": "user", "content": "Say OK."}], 8)
    rows = query_gpu()
    rec["vram_after_warmup_mb"] = rows[0]["vram_used_mb"] if rows else None
    peak = rec["vram_after_warmup_mb"] or 0.0

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

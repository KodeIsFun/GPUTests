"""W2 "Bake-off": the same torch + whisper-tiny work on every free GPU we reach.

One deliberately platform-agnostic file (no lab/ imports) pushed identically to
Kaggle T4, Kaggle P100, and run via `colab run --gpu T4`, so wall-time
differences are the platform's, not the script's.

Measures, in order:
  1. hardware identity (nvidia-smi) — which card did the free tier hand us?
  2. fp16 matmul throughput — same constants as W0/1's hello-t4, so the two
     weeks' numbers are directly comparable;
  3. `pip install openai-whisper` seconds;
  4. whisper-tiny cold load (includes the ~72 MB weight download on a fresh VM);
  5. transcribe() wall time on a fixed 30 s synthetic waveform -> RTF.

Writes timings.json (schema: {gpu, vram_mb, wall_s, metric, git_sha, platform})
and echoes it between ###TIMINGS### markers so a Colab stdout capture can
extract it without filesystem access.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Stamped at push time with the commit this job was built from.
GIT_SHA: str | None = "1da4ed4"

MATMUL_SIZE = 4096
MATMUL_ITERS = 20
PIP_PACKAGE = "openai-whisper"
WHISPER_MODEL = "tiny"
AUDIO_S = 30.0
SAMPLE_RATE = 16000

GPU_QUERY = "name,memory.total,driver_version,compute_cap"


def output_dir() -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.exists():
        return kaggle
    colab = Path("/content")
    if colab.exists():
        return colab
    fallback = Path("results")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def platform() -> str:
    if Path("/kaggle/working").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        return "colab"
    return "local"


def run(cmd: list[str], timeout: int = 600) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def query_gpu() -> dict:
    code, raw = run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        timeout=120,
    )
    empty = {"name": None, "memory_mb": None, "driver": None, "compute_cap": None}
    if code != 0 or not raw.strip():
        return empty
    fields = [part.strip() for part in raw.strip().splitlines()[0].split(",")]
    if len(fields) < 4:
        return empty
    try:
        memory_mb = int(float(fields[1]))
    except ValueError:
        memory_mb = None
    return {
        "name": fields[0] or None,
        "memory_mb": memory_mb,
        "driver": fields[2] or None,
        "compute_cap": fields[3] or None,
    }


def cuda_version(smi: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([\d.]+)", smi)
    return match.group(1) if match else None


def matmul_gflops(size: int = MATMUL_SIZE, iters: int = MATMUL_ITERS) -> float | None:
    """fp16 matmul throughput — the one real number on this card."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    dtype = torch.float16
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    try:
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
    except RuntimeError:
        return None
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        return None
    return round(2.0 * size**3 * iters / elapsed / 1e9, 1)


def torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.__version__


def pip_install_seconds(package: str = PIP_PACKAGE) -> float | None:
    started = time.perf_counter()
    code, _ = run(
        [sys.executable, "-m", "pip", "install", "--quiet", package],
        timeout=900,
    )
    if code != 0:
        return None
    return round(time.perf_counter() - started, 2)


def synthetic_audio(seconds: float = AUDIO_S, sr: int = SAMPLE_RATE):
    """Fixed speech-like waveform so every platform transcribes identical input.

    Formant-ish sweeping sines under an amplitude-modulated envelope: whisper
    will hallucinate words at it, which is fine — the claim is wall time, never
    transcript quality.
    """
    import numpy as np

    rng = np.random.default_rng(20260919)
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 2.5 * t)) ** 2
    wave = np.zeros_like(t)
    for base in (140.0, 720.0, 1220.0, 2600.0):
        f0 = base * (1.0 + 0.08 * np.sin(2.0 * np.pi * 0.7 * t))
        wave += np.sin(2.0 * np.pi * f0 * t) / base**0.5
    wave += 0.05 * rng.standard_normal(t.shape)
    audio = (wave * envelope).astype(np.float32)
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


def whisper_block() -> dict:
    """pip -> cold load -> transcribe, each timed, failures recorded not fatal."""
    block: dict = {
        "pip_install_s": pip_install_seconds(),
        "pip_package": PIP_PACKAGE,
        "model": WHISPER_MODEL,
    }
    try:
        import whisper
    except ImportError:
        block["error"] = "whisper import failed after pip install"
        return block

    started = time.perf_counter()
    try:
        model = whisper.load_model(WHISPER_MODEL)
    except Exception as exc:  # download or load failure is itself a data point
        block["load_s"] = round(time.perf_counter() - started, 2)
        block["error"] = f"load_model failed: {type(exc).__name__}: {exc}"
        return block
    block["load_s"] = round(time.perf_counter() - started, 2)

    audio = synthetic_audio()
    started = time.perf_counter()
    try:
        result = model.transcribe(audio)
    except Exception as exc:
        block["transcribe_s"] = round(time.perf_counter() - started, 2)
        block["error"] = f"transcribe failed: {type(exc).__name__}: {exc}"
        return block
    block["transcribe_s"] = round(time.perf_counter() - started, 2)
    block["rtf"] = round(AUDIO_S / block["transcribe_s"], 2)
    block["transcript_chars"] = len(result.get("text", ""))
    # Evidence only — never a quality claim; the input is synthetic tones.
    block["transcript_head"] = (result.get("text", "") or "")[:120]
    return block


def main() -> None:
    started = time.perf_counter()
    out = output_dir()

    code, smi = run(["nvidia-smi"], timeout=120)
    if code != 0:
        smi = f"nvidia-smi failed:\n{smi}"
    (out / "nvidia-smi.txt").write_text(smi)

    gpu = query_gpu()
    details = {
        "driver": gpu["driver"],
        "compute_cap": gpu["compute_cap"],
        "cuda_version": cuda_version(smi),
        "torch": torch_version(),
        "python": sys.version.split()[0],
        "matmul": f"fp16 {MATMUL_SIZE}x{MATMUL_SIZE} x{MATMUL_ITERS}",
        "metric_unit": "GFLOP/s",
        "whisper": whisper_block(),
    }
    payload = {
        "gpu": gpu["name"],
        "vram_mb": gpu["memory_mb"],
        # Computed after the whisper block so wall_s spans the whole job.
        # (W2's three runs measured it before the whisper block — comparable
        # across lanes, but pre-whisper only.)
        "wall_s": round(time.perf_counter() - started, 3),
        "metric": matmul_gflops(),
        "git_sha": GIT_SHA,
        "platform": platform(),
        "details": details,
    }
    (out / "timings.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("###TIMINGS###")
    print(json.dumps(payload, indent=2))
    print("###TIMINGS###")


if __name__ == "__main__":
    main()

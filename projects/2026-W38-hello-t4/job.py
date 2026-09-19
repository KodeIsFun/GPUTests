"""W0/1 "Hello T4": what hardware does a free GPU kernel actually hand us?

Deliberately platform-agnostic so W2 can diff Kaggle against Colab with the
same file. Runs remotely — must not import anything from lab/.

Writes timings.json with the schema PLAN.md requires:
    {gpu, vram_mb, wall_s, metric, git_sha, platform}
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
GIT_SHA: str | None = None

MATMUL_SIZE = 4096
MATMUL_ITERS = 20
PIP_PACKAGE = "tabulate"

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
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", package],
        timeout=600,
    )
    if code != 0:
        return None
    return round(time.perf_counter() - started, 2)


def main() -> None:
    started = time.perf_counter()
    out = output_dir()

    code, smi = run(["nvidia-smi"], timeout=120)
    if code != 0:
        smi = f"nvidia-smi failed:\n{smi}"
    (out / "nvidia-smi.txt").write_text(smi)

    gpu = query_gpu()
    payload = {
        "gpu": gpu["name"],
        "vram_mb": gpu["memory_mb"],
        "wall_s": round(time.perf_counter() - started, 3),
        "metric": matmul_gflops(),
        "git_sha": GIT_SHA,
        "platform": platform(),
        "details": {
            "driver": gpu["driver"],
            "compute_cap": gpu["compute_cap"],
            "cuda_version": cuda_version(smi),
            "torch": torch_version(),
            "python": sys.version.split()[0],
            "pip_install_s": pip_install_seconds(),
            "pip_package": PIP_PACKAGE,
            "matmul": f"fp16 {MATMUL_SIZE}x{MATMUL_SIZE} x{MATMUL_ITERS}",
            "metric_unit": "GFLOP/s",
        },
    }
    (out / "timings.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

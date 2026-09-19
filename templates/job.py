"""Kaggle script kernel. Writes results/timings.json. Do not import lab.* on Kaggle."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

OUT = Path("/kaggle/working")
if not OUT.exists():
    OUT = Path("results")
    OUT.mkdir(parents=True, exist_ok=True)


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"nvidia-smi failed: {exc}"


def gpu_name(smi: str) -> str:
    for line in smi.splitlines():
        for marker in ("Tesla T4", "Tesla P100", "Tesla L4", "T4", "P100", "L4"):
            if marker in line:
                return marker if marker.startswith("Tesla") else line.strip()
    return "unknown"


def main() -> None:
    started = time.time()
    smi = nvidia_smi()
    (OUT / "nvidia-smi.txt").write_text(smi)
    payload = {
        "gpu": gpu_name(smi),
        "vram_mb": None,
        "wall_s": round(time.time() - started, 3),
        "metric": None,
        "git_sha": None,
        "platform": "kaggle",
    }
    (OUT / "timings.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

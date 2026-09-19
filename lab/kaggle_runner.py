from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from lab.quota import GPU_JOB_MAX_HOURS, BudgetError, GpuQuota, require_gpu_job

DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class KernelSpec:
    username: str
    slug: str
    title: str
    code_file: str
    budget_hours: float
    kernel_type: str = "script"
    language: str = "python"


def default_kaggle_bin() -> Path:
    return ROOT / ".venv" / "bin" / "kaggle"


def write_kernel_metadata(
    folder: Path, spec: KernelSpec, quota: GpuQuota | None = None
) -> Path:
    if quota is not None:
        require_gpu_job(quota, spec.budget_hours)
    elif spec.budget_hours > GPU_JOB_MAX_HOURS:
        raise BudgetError(
            f"budget_hours {spec.budget_hours} exceeds {GPU_JOB_MAX_HOURS}"
        )
    payload = {
        "id": f"{spec.username}/{spec.slug}",
        "title": spec.title,
        "code_file": spec.code_file,
        "language": spec.language,
        "kernel_type": spec.kernel_type,
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "keywords": ["gpu", "free-gpu-lab"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    path = folder / "kernel-metadata.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def build_push_command(
    folder: Path,
    kaggle_bin: str,
    quota: GpuQuota,
    budget_hours: float,
) -> list[str]:
    require_gpu_job(quota, budget_hours)
    return [
        kaggle_bin,
        "kernels",
        "push",
        "-p",
        str(folder),
        "--accelerator",
        DEFAULT_ACCELERATOR,
    ]


def build_status_command(kaggle_bin: str, username: str, slug: str) -> list[str]:
    return [kaggle_bin, "kernels", "status", f"{username}/{slug}"]


def build_output_command(kaggle_bin: str, username: str, slug: str) -> list[str]:
    return [kaggle_bin, "kernels", "output", f"{username}/{slug}"]

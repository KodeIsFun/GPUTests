from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

from lab.quota import GPU_JOB_MAX_HOURS, BudgetError, GpuQuota, require_gpu_job

DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
ROOT = Path(__file__).resolve().parent.parent


class SlugMismatch(ValueError):
    pass


def kaggle_slugify(title: str) -> str:
    """Kaggle derives the kernel slug from the title, not from the metadata id.

    'Free GPU bake-off (P100)' -> 'free-gpu-bake-off-p100'. Pushing with an id
    that disagrees creates the kernel under the title slug and then 409s on
    every later push of that id (W2, learned 2026-09-19).
    """
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


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
    derived = kaggle_slugify(spec.title)
    if derived != spec.slug:
        raise SlugMismatch(
            f"slug {spec.slug!r} does not resolve from title {spec.title!r} "
            f"(expected {derived!r}); Kaggle names the kernel after the title "
            "and a mismatched id makes every later push 409"
        )
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
    accelerator: str = DEFAULT_ACCELERATOR,
) -> list[str]:
    require_gpu_job(quota, budget_hours)
    # -t makes budget_hours binding at the platform level instead of merely
    # advisory: Kaggle kills the run at the cap the same way it would at its
    # own session limit.
    return [
        kaggle_bin,
        "kernels",
        "push",
        "-p",
        str(folder),
        "--accelerator",
        accelerator,
        "-t",
        str(int(budget_hours * 3600)),
    ]


def build_status_command(kaggle_bin: str, username: str, slug: str) -> list[str]:
    return [kaggle_bin, "kernels", "status", f"{username}/{slug}"]


def build_output_command(kaggle_bin: str, username: str, slug: str) -> list[str]:
    return [kaggle_bin, "kernels", "output", f"{username}/{slug}"]

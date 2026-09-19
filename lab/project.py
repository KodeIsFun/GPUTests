from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil

from lab.kaggle_runner import KernelSpec, write_kernel_metadata

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"


@dataclass(frozen=True)
class WeekProject:
    year_week: str
    slug: str
    title: str
    budget_hours: float
    estimated_usd: float
    username: str = "emdadh"


def scaffold(project: WeekProject, projects_dir: Path | None = None) -> Path:
    dest = (projects_dir or ROOT / "projects") / f"{project.year_week}-{project.slug}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results").mkdir(exist_ok=True)
    (dest / "tweets").mkdir(exist_ok=True)
    for name in ("job.py", "task.py", "notebook.ipynb"):
        src = TEMPLATES / name
        if src.exists():
            shutil.copy(src, dest / name)
    write_kernel_metadata(
        dest,
        KernelSpec(
            username=project.username,
            slug=project.slug,
            title=project.title,
            code_file="job.py",
            budget_hours=project.budget_hours,
        ),
    )
    manifest = {
        "title": project.title,
        "slug": project.slug,
        "budget_hours": project.budget_hours,
        "estimated_usd": project.estimated_usd,
        "hypothesis": "",
        "success_metric": "",
    }
    (dest / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dest

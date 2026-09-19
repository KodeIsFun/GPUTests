"""Every scheduled task file must be lint-clean before it can be pushed.

Deliberately AST-only: importing kaggle_benchmarks is slow and must not be a
precondition for running the unit suite. Use `python -m lab.task_check` for the
deeper, still-free pre-flight that actually imports the task.

A week's folder is named for its GPU kernel (`hello-t4`) while its eval task
carries its own slug (`hello-proxy`), so folder name and task name are checked
independently rather than assumed equal.
"""
from pathlib import Path

import pytest

from lab.benchmarks_runner import declared_task_names, lint_task_source

ROOT = Path(__file__).resolve().parent.parent
TASK_FILES = sorted(ROOT.glob("projects/*/task.py"))
IDS = [p.parent.name for p in TASK_FILES]


def test_project_layout_is_discoverable() -> None:
    assert (ROOT / "projects").is_dir()


@pytest.mark.parametrize("path", TASK_FILES, ids=IDS)
def test_task_file_declares_a_task(path: Path) -> None:
    assert declared_task_names(path.read_text()), f"{path} declares no @kbench.task"


@pytest.mark.parametrize("path", TASK_FILES, ids=IDS)
def test_every_declared_slug_passes_lint(path: Path) -> None:
    source = path.read_text()
    for name in declared_task_names(source):
        lint_task_source(source, name)


@pytest.mark.parametrize("path", TASK_FILES, ids=IDS)
def test_task_file_cannot_bypass_the_model_proxy(path: Path) -> None:
    """PLAN.md: no proxy call outside a published @kbench.task."""
    source = path.read_text()
    for forbidden in ("import openai", "import anthropic", "requests.post"):
        assert forbidden not in source, f"{path} appears to bypass the Model Proxy"

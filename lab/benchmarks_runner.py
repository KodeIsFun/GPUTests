from __future__ import annotations

from pathlib import Path
import ast

from lab.quota import BudgetError, ProxyQuota, require_proxy_run

MAX_MODELS_PER_RUN = 2
ROOT = Path(__file__).resolve().parent.parent


class TaskLintError(ValueError):
    pass


def default_kaggle_bin() -> Path:
    return ROOT / ".venv" / "bin" / "kaggle"


def declared_task_names(source: str) -> list[str]:
    """The @kbench.task(name=...) slugs a task file declares.

    Deliberately independent of the project folder name: a week's folder is
    named for its GPU kernel (hello-t4) while its eval task carries its own
    slug (hello-proxy).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise TaskLintError(f"task file is not valid Python: {exc}") from exc
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "task":
            for keyword in node.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    names.append(str(keyword.value.value))
    return names


def has_execution_call(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise TaskLintError(f"task file is not valid Python: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"run", "evaluate"}:
            return True
    return False


def lint_task_source(source: str, slug: str) -> None:
    names = declared_task_names(source)
    if slug not in names:
        raise TaskLintError(
            f"task slug {slug!r} does not match any @kbench.task name {names}"
        )
    if not has_execution_call(source):
        raise TaskLintError(
            "task file has no .run() or .evaluate() call — push would be a silent no-op"
        )


def split_models_for_daily_budget(
    models: list[str], per_day: int = MAX_MODELS_PER_RUN
) -> list[list[str]]:
    chunk = min(per_day, MAX_MODELS_PER_RUN)
    if chunk < 1:
        raise ValueError("per_day must be >= 1")
    return [models[i : i + chunk] for i in range(0, len(models), chunk)]


def build_run_command(
    kaggle_bin: str,
    slug: str,
    models: list[str],
    quota: ProxyQuota,
    estimated_usd: float,
    publish_day: bool = False,
) -> list[str]:
    if not models:
        raise BudgetError("at least one model required")
    if len(models) > MAX_MODELS_PER_RUN:
        raise BudgetError(
            f"{len(models)} models in one run; max {MAX_MODELS_PER_RUN} "
            "(split with split_models_for_daily_budget)"
        )
    require_proxy_run(quota, estimated_usd, publish_day=publish_day)
    cmd = [kaggle_bin, "b", "t", "run", slug]
    for model in models:
        cmd.extend(["-m", model])
    cmd.append("--wait")
    return cmd


def build_push_command(kaggle_bin: str, slug: str, task_path: Path) -> list[str]:
    lint_task_source(task_path.read_text(), slug)
    return [kaggle_bin, "b", "t", "push", slug, "-f", str(task_path), "--wait"]

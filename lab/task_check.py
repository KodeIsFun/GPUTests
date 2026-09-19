"""Offline pre-flight for a Benchmarks task file — run this BEFORE spending Proxy $.

`lint_task_source` catches the silent no-op (a missing `.run()` / `.evaluate()`)
by reading the AST. This goes further: it actually imports the task, so a bad
`@kbench.task` name, an unregistered return annotation, or an `evaluate()` whose
grid cannot bind its parameters fails here instead of on Kaggle.

Uses no model: `Task.run` / `Task.evaluate` are stubbed out, so nothing reaches
the Model Proxy and nothing is billed.

    .venv/bin/python -m lab.task_check projects/2026-W38-hello-t4/task.py
"""
from __future__ import annotations

from pathlib import Path
import contextlib
import inspect
import sys
import tempfile

from lab.benchmarks_runner import declared_task_names, lint_task_source


def check(path: Path) -> list[str]:
    source = path.read_text()
    names = declared_task_names(source)
    if not names:
        raise SystemExit(f"{path}: declares no @kbench.task")
    for name in names:
        lint_task_source(source, name)

    from kaggle_benchmarks import tasks

    captured: list[tuple[str, dict]] = []
    tasks.Task.evaluate = lambda self, *a, **k: captured.append(
        ("evaluate", {"task": self, **k})
    )
    tasks.Task.run = lambda self, *a, **k: captured.append(
        ("run", {"task": self, "args": a, **k})
    )

    namespace: dict = {"__name__": "__main__", "__file__": str(path)}
    # dont_inherit=True is load-bearing: without it this module's own
    # `from __future__ import annotations` leaks into the task file, turning
    # its `-> bool` annotation into the string "bool", which kbench cannot
    # match against its registered result types.
    code = compile(source, str(path), "exec", dont_inherit=True)
    # kbench writes <task>-*.run.json into the CWD as runs are recorded. Run
    # from a scratch directory so a pre-flight never litters the repo.
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
        exec(code, namespace)

    if not captured:
        raise SystemExit(
            f"{path}: no .run()/.evaluate() executed at import time — "
            "on Kaggle this would be a silent no-op"
        )

    lines = [f"{path}  (declares {names})"]
    for kind, call in captured:
        task = call["task"]
        lines.append(f"  {kind}: {task.name!r}")
        lines.append(f"    result type : {task.result_type.__name__}")
        lines.append(f"    signature   : {inspect.signature(task.func)}")
        if kind == "evaluate":
            grid = call.get("grid") or {}
            data = call.get("evaluation_data")
            lines.append(f"    grid keys   : {list(grid)}")
            if data is not None:
                lines.append(
                    f"    rows        : {len(data)} cols={list(data.columns)}"
                )
            lines.append(f"    on_failure  : {call.get('on_failure')}")
            missing = set(inspect.signature(task.func).parameters) - set(grid)
            if data is not None:
                missing -= set(data.columns)
            missing.discard("llm")
            if missing:
                lines.append(
                    f"    WARNING     : params never bound by grid or data: "
                    f"{sorted(missing)}"
                )
    return lines


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]]
    if not targets:
        targets = sorted(Path("projects").glob("*/task.py"))
    if not targets:
        print("no task files found", file=sys.stderr)
        return 1
    for path in targets:
        for line in check(path):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

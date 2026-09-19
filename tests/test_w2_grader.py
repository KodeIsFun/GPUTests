"""The W2 emit-kernel-metadata grader, tested offline.

The task file cannot be imported (its module-level .evaluate() would reach the
Model Proxy), so the pure grading functions are lifted out of its AST and
executed on their own. This keeps the grader under test without spending a
nanodollar or importing kaggle_benchmarks.
"""
import ast
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "projects" / "2026-W38-bakeoff-t4" / "task.py"

GRADED_FUNCS = {"extract_json_object", "parse_expect", "grade"}


def load_grader():
    tree = ast.parse(TASK.read_text())
    # The lifted function bodies use the module's json/re imports; provide
    # them since the AST extraction pulls FunctionDefs only.
    namespace: dict = {"json": json, "re": re}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in GRADED_FUNCS:
            exec(compile(ast.Module([node], []), str(TASK), "exec"), namespace)
    missing = GRADED_FUNCS - namespace.keys()
    assert not missing, f"task file no longer defines {missing}"
    return namespace["grade"]


@pytest.fixture(scope="module")
def grade():
    return load_grader()


FULL_META = """
Here is the metadata you asked for:

```json
{
  "id": "emdadh/t4-demo",
  "title": "T4 Demo",
  "code_file": "job.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true
}
```
"""

GOOD_EXPECT = (
    "id=emdadh/t4-demo;title=T4 Demo;kernel_type=script;language=python;"
    "enable_gpu=true;enable_internet=true;code_file=job.py"
)


def test_full_metadata_passes(grade) -> None:
    assert grade("metadata", GOOD_EXPECT, FULL_META) is True


def test_string_true_fails_strict_boolean(grade) -> None:
    # Kaggle's schema wants real JSON booleans; "true" as a string would not
    # push, so the grader must fail it.
    liar = FULL_META.replace('"enable_gpu": true', '"enable_gpu": "true"')
    assert grade("metadata", GOOD_EXPECT, liar) is False


def test_missing_field_fails(grade) -> None:
    amputee = FULL_META.replace('"kernel_type": "script",\n', "")
    assert grade("metadata", GOOD_EXPECT, amputee) is False


def test_bare_json_without_fence_passes(grade) -> None:
    assert grade("metadata", GOOD_EXPECT, FULL_META.replace("```json\n", "").replace("```", ""))


def test_no_json_fails(grade) -> None:
    assert grade("metadata", GOOD_EXPECT, "just use kaggle kernels push") is False


def test_private_flip_checked(grade) -> None:
    assert grade("metadata", "id=emdadh/t4-demo-private;is_private=true", FULL_META.replace(
        '"id": "emdadh/t4-demo"', '"id": "emdadh/t4-demo-private"'
    )) is False  # is_private still false


def test_key_question(grade) -> None:
    assert grade("key", "enable_gpu", "The key is enable_gpu.") is True
    assert grade("key", "enable_gpu", "use the accelerator flag") is False


def test_seconds_arithmetic(grade) -> None:
    assert grade("seconds", "5400", "5400") is True
    assert grade("seconds", "5400", "That's 5,400 seconds.") is True
    assert grade("seconds", "5400", "5400 s") is True
    assert grade("seconds", "5400", "about 54000 seconds") is False
    assert grade("seconds", "5400", "5400.5") is False
    assert grade("seconds", "28800", "8 hours = 28800 seconds") is True


def test_unknown_kind_fails_closed(grade) -> None:
    assert grade("essay", "anything", "anything") is False

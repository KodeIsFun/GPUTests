import pytest

from pathlib import Path

from lab.benchmarks_runner import (
    TaskLintError,
    build_push_command,
    build_run_command,
    lint_task_source,
    split_models_for_daily_budget,
)
from lab.quota import BudgetError, ProxyQuota

AMPLE_PROXY = ProxyQuota(daily_remaining=10.0, monthly_remaining=100.0)
ROOT = Path(__file__).resolve().parent.parent


VALID_TASK = '''
# %%
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="hello-proxy")
def hello_proxy(llm):
    response = llm.prompt("What is 2 + 2?")
    kbench.assertions.assert_in("4", response, expectation="contains 4")

hello_proxy.run(kbench.llm)
'''


def test_accept_task_with_decorator_and_run() -> None:
    lint_task_source(VALID_TASK, slug="hello-proxy")


def test_reject_missing_run_call() -> None:
    source = VALID_TASK.replace("hello_proxy.run(kbench.llm)", "")
    with pytest.raises(TaskLintError, match="run"):
        lint_task_source(source, slug="hello-proxy")


def test_commented_run_does_not_count() -> None:
    source = VALID_TASK.replace(
        "hello_proxy.run(kbench.llm)",
        "# hello_proxy.run(kbench.llm)",
    )
    with pytest.raises(TaskLintError, match="run"):
        lint_task_source(source, slug="hello-proxy")


def test_template_task_lints() -> None:
    source = (ROOT / "templates" / "task.py").read_text()
    lint_task_source(source, slug="hello-proxy")


def test_reject_slug_mismatch() -> None:
    with pytest.raises(TaskLintError, match="slug"):
        lint_task_source(VALID_TASK, slug="other-task")


def test_split_models_two_per_day() -> None:
    batches = split_models_for_daily_budget(
        ["flash", "haiku", "qwen", "deepseek", "sonnet"]
    )
    assert batches == [
        ["flash", "haiku"],
        ["qwen", "deepseek"],
        ["sonnet"],
    ]


def test_split_clamps_per_day_above_two() -> None:
    batches = split_models_for_daily_budget(["a", "b", "c"], per_day=9)
    assert all(len(batch) <= 2 for batch in batches)


def test_run_command_repeats_dash_m() -> None:
    cmd = build_run_command(
        "/opt/kaggle",
        "hello-proxy",
        ["flash", "haiku"],
        AMPLE_PROXY,
        estimated_usd=2.0,
    )
    assert cmd == [
        "/opt/kaggle",
        "b",
        "t",
        "run",
        "hello-proxy",
        "-m",
        "flash",
        "-m",
        "haiku",
        "--wait",
    ]


def test_run_command_refuses_three_models() -> None:
    with pytest.raises(BudgetError, match="3 models"):
        build_run_command(
            "/opt/kaggle",
            "hello-proxy",
            ["flash", "haiku", "qwen"],
            AMPLE_PROXY,
            estimated_usd=2.0,
        )


def test_push_lints_file(tmp_path: Path) -> None:
    path = tmp_path / "task.py"
    path.write_text(VALID_TASK)
    cmd = build_push_command("/opt/kaggle", "hello-proxy", path)
    assert "-f" in cmd
    assert str(path) in cmd

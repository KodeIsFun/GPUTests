from pathlib import Path

import pytest

from lab.kaggle_runner import (
    KernelSpec,
    build_push_command,
    write_kernel_metadata,
)
from lab.quota import BudgetError, GpuQuota

AMPLE = GpuQuota(used_hours=0.0, remaining_hours=30.0, total_hours=30.0)


def test_metadata_is_public_gpu_script(tmp_path: Path) -> None:
    spec = KernelSpec(
        username="emdadh",
        slug="hello-t4",
        title="Hello T4",
        code_file="job.py",
        budget_hours=1,
    )
    path = write_kernel_metadata(tmp_path, spec)
    data = path.read_text()
    assert '"enable_gpu": true' in data
    assert '"enable_internet": true' in data
    assert '"is_private": false' in data
    assert "emdadh/hello-t4" in data


def test_push_command_requests_t4(tmp_path: Path) -> None:
    cmd = build_push_command(tmp_path, "/opt/kaggle", AMPLE, 1)
    assert cmd == [
        "/opt/kaggle",
        "kernels",
        "push",
        "-p",
        str(tmp_path),
        "--accelerator",
        "NvidiaTeslaT4",
    ]


def test_push_refuses_when_week_hard_stop_would_trip(tmp_path: Path) -> None:
    tight = GpuQuota(used_hours=22.0, remaining_hours=8.0, total_hours=30.0)
    with pytest.raises(BudgetError):
        build_push_command(tmp_path, "/opt/kaggle", tight, 4)


def test_refuse_to_build_metadata_for_too_long_job(tmp_path: Path) -> None:
    spec = KernelSpec(
        username="emdadh",
        slug="too-long",
        title="Too Long",
        code_file="job.py",
        budget_hours=10,
    )
    with pytest.raises(BudgetError):
        write_kernel_metadata(tmp_path, spec)

from pathlib import Path

import pytest

from lab.kaggle_runner import (
    KernelSpec,
    build_push_command,
    kaggle_slugify,
    write_kernel_metadata,
)
from lab.quota import GpuQuota

AMPLE = GpuQuota(used_hours=0.0, remaining_hours=30.0, total_hours=30.0)


def test_push_command_can_request_p100(tmp_path: Path) -> None:
    """W2 bake-off: same folder, same budget, different card."""
    cmd = build_push_command(
        tmp_path, "/opt/kaggle", AMPLE, 1, accelerator="NvidiaTeslaP100"
    )
    assert cmd[cmd.index("--accelerator") + 1] == "NvidiaTeslaP100"
    assert cmd[cmd.index("-t") + 1] == "3600"


def test_push_command_defaults_to_t4(tmp_path: Path) -> None:
    cmd = build_push_command(tmp_path, "/opt/kaggle", AMPLE, 1)
    assert cmd[cmd.index("--accelerator") + 1] == "NvidiaTeslaT4"


def test_slugify_matches_kaggle_title_resolution() -> None:
    assert kaggle_slugify("Hello T4") == "hello-t4"
    assert kaggle_slugify("Free GPU bake-off (P100)") == "free-gpu-bake-off-p100"
    assert kaggle_slugify("Free GPU bake-off") == "free-gpu-bake-off"


def test_metadata_refuses_title_slug_mismatch(tmp_path: Path) -> None:
    """The W2 409: id 'bakeoff-p100' vs title 'Free GPU bake-off (P100)'."""
    with pytest.raises(ValueError, match="free-gpu-bake-off-p100"):
        write_kernel_metadata(
            tmp_path,
            KernelSpec(
                username="emdadh",
                slug="bakeoff-p100",
                title="Free GPU bake-off (P100)",
                code_file="job.py",
                budget_hours=1,
            ),
        )

from __future__ import annotations

from pathlib import Path
import shutil

FORBIDDEN = frozenset({"auth", "console", "drivemount", "repl"})

# `colab run` defaults --timeout to 30s, which kills any real benchmark.
# PLAN.md keeps Colab jobs under 90 minutes.
DEFAULT_TIMEOUT_S = 5400.0


class ColabError(ValueError):
    pass


def default_colab_bin() -> Path:
    found = shutil.which("colab")
    if found is not None:
        return Path(found)
    return Path.home() / ".local" / "bin" / "colab"


def assert_safe_subcommand(subcommand: str) -> None:
    if subcommand in FORBIDDEN:
        raise ColabError(
            f"colab {subcommand} is interactive and must not be run by the agent"
        )


def build_ephemeral_run(
    script: str,
    session: str,
    config_path: str,
    gpu: str = "T4",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    colab_bin: str | None = None,
) -> list[str]:
    assert_safe_subcommand("run")
    if timeout_s <= 0:
        raise ColabError("timeout_s must be > 0")
    return [
        str(colab_bin or default_colab_bin()),
        "--config",
        config_path,
        "run",
        "--gpu",
        gpu,
        "-s",
        session,
        "--timeout",
        str(timeout_s),
        script,
    ]


def build_stop(
    session: str, config_path: str, colab_bin: str | None = None
) -> list[str]:
    return [
        str(colab_bin or default_colab_bin()),
        "--config",
        config_path,
        "stop",
        "-s",
        session,
    ]


def ephemeral_commands(
    script: str,
    session: str,
    config_path: str,
    gpu: str = "T4",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    colab_bin: str | None = None,
) -> tuple[list[str], list[str]]:
    return (
        build_ephemeral_run(
            script, session, config_path, gpu, timeout_s, colab_bin
        ),
        build_stop(session, config_path, colab_bin),
    )

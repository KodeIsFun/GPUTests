import pytest

from lab.colab_runner import (
    DEFAULT_TIMEOUT_S,
    ColabError,
    assert_safe_subcommand,
    build_ephemeral_run,
    build_stop,
    ephemeral_commands,
)


def test_ephemeral_run_self_cleans() -> None:
    cmd = build_ephemeral_run(
        script="job.py",
        session="hello-t4",
        config_path="/tmp/hello.json",
        gpu="T4",
        colab_bin="/opt/colab",
    )
    assert cmd[0] == "/opt/colab"
    assert "--config" in cmd
    assert "/tmp/hello.json" in cmd
    assert "run" in cmd
    assert "--gpu" in cmd
    assert "T4" in cmd
    assert "--keep" not in cmd
    assert "job.py" in cmd


def test_run_overrides_thirty_second_cli_default() -> None:
    cmd = build_ephemeral_run(
        script="job.py",
        session="hello-t4",
        config_path="/tmp/hello.json",
        colab_bin="/opt/colab",
    )
    assert "--timeout" in cmd
    assert cmd[cmd.index("--timeout") + 1] == str(DEFAULT_TIMEOUT_S)
    assert DEFAULT_TIMEOUT_S > 30.0


def test_run_accepts_explicit_timeout() -> None:
    cmd = build_ephemeral_run(
        script="job.py",
        session="hello-t4",
        config_path="/tmp/hello.json",
        timeout_s=120.0,
        colab_bin="/opt/colab",
    )
    assert cmd[cmd.index("--timeout") + 1] == "120.0"


def test_run_refuses_non_positive_timeout() -> None:
    with pytest.raises(ColabError):
        build_ephemeral_run(
            script="job.py",
            session="hello-t4",
            config_path="/tmp/hello.json",
            timeout_s=0,
            colab_bin="/opt/colab",
        )


def test_script_follows_timeout_so_args_are_not_swallowed() -> None:
    cmd = build_ephemeral_run(
        script="job.py",
        session="hello-t4",
        config_path="/tmp/hello.json",
        colab_bin="/opt/colab",
    )
    assert cmd.index("job.py") == cmd.index("--timeout") + 2


def test_stop_always_names_session() -> None:
    cmd = build_stop("hello-t4", "/tmp/hello.json", colab_bin="/opt/colab")
    assert "stop" in cmd
    assert "-s" in cmd
    assert "hello-t4" in cmd


@pytest.mark.parametrize(
    "subcommand", ["auth", "console", "drivemount", "repl"]
)
def test_forbidden_interactive_commands(subcommand: str) -> None:
    with pytest.raises(ColabError):
        assert_safe_subcommand(subcommand)


def test_ephemeral_commands_pair_run_with_stop() -> None:
    run_cmd, stop_cmd = ephemeral_commands(
        "job.py", "hello-t4", "/tmp/hello.json", colab_bin="/opt/colab"
    )
    assert "run" in run_cmd
    assert "stop" in stop_cmd
    assert "hello-t4" in stop_cmd
    assert run_cmd[0] == stop_cmd[0]

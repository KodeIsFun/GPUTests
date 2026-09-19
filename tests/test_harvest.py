"""The harvester is what makes a per-model cost claim citable, so test the
flattening and the arithmetic on a synthetic run tree."""
from pathlib import Path
import json

import pytest

from lab.harvest import read_runs, summarize, write_eval_results


def _run(in_tok, out_tok, in_nano, out_nano, latency_ms, passed=True):
    return {
        "conversations": [
            {
                "metrics": {
                    "inputTokens": in_tok,
                    "outputTokens": out_tok,
                    "inputTokensCostNanodollars": str(in_nano),
                    "outputTokensCostNanodollars": str(out_nano),
                    "totalBackendLatencyMs": str(latency_ms),
                }
            }
        ],
        "assertions": (
            [{"status": "BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED"}]
            if passed
            else [{"status": "BENCHMARK_TASK_RUN_ASSERTION_STATUS_FAILED"}]
        ),
    }


def _write_tree(root: Path) -> Path:
    """Mimic <out>/<task>/<version>/<model>/<session>/<file>.run.json"""
    for model, records in {
        "cheap-model": [_run(20, 1, 8400, 2500, 347), _run(20, 1, 8400, 2500, 400)],
        "dear-model": [_run(20, 100, 8400, 900000, 1200)],
    }.items():
        d = root / "hello-proxy" / "1" / model / "12345"
        d.mkdir(parents=True)
        for i, record in enumerate(records):
            (d / f"run_{i}.run.json").write_text(json.dumps(record))
    return root / "hello-proxy"


def test_nanodollars_become_usd(tmp_path: Path) -> None:
    runs = read_runs(_write_tree(tmp_path))
    assert runs["cheap-model"][0]["input_tokens_cost"] == pytest.approx(8400 / 1e9)


def test_summarize_counts_passes_and_tokens(tmp_path: Path) -> None:
    runs = read_runs(_write_tree(tmp_path))
    summary = summarize({"cheap-model": runs["cheap-model"]})
    assert summary["items"] == 2
    assert summary["passed"] == 2
    assert summary["score"] == 1.0
    assert summary["output_tokens"] == 2
    assert summary["output_tokens_per_item"] == 1.0


def test_failed_assertion_is_not_a_pass(tmp_path: Path) -> None:
    d = tmp_path / "t" / "1" / "m" / "9"
    d.mkdir(parents=True)
    (d / "r.run.json").write_text(json.dumps(_run(1, 1, 1, 1, 1, passed=False)))
    summary = summarize(read_runs(tmp_path / "t"))
    assert summary["passed"] == 0
    assert summary["score"] == 0.0


def _bool_run(in_tok, out_tok, in_nano, out_nano, latency_ms, passed: bool):
    """W2 shape: a Boolean-result task has null assertions and an aggregated
    results[].booleanResult instead."""
    run = _run(in_tok, out_tok, in_nano, out_nano, latency_ms, passed=True)
    run["assertions"] = None
    run["results"] = [{"type": "AGGREGATED", "booleanResult": passed}]
    return run


def test_boolean_result_task_is_scored(tmp_path: Path) -> None:
    d = tmp_path / "t" / "1" / "m" / "9"
    d.mkdir(parents=True)
    (d / "y.run.json").write_text(
        json.dumps(_bool_run(100, 300, 1, 2, 700, passed=True))
    )
    (d / "n.run.json").write_text(
        json.dumps(_bool_run(100, 300, 1, 2, 700, passed=False))
    )
    summary = summarize(read_runs(tmp_path / "t"))
    assert summary["passed"] == 1
    assert summary["score"] == 0.5


def test_boolean_result_false_is_not_vacuously_true(tmp_path: Path) -> None:
    # The W2 scorer bug: all([]) on null assertions called every item a pass.
    d = tmp_path / "t" / "1" / "m" / "9"
    d.mkdir(parents=True)
    (d / "n.run.json").write_text(
        json.dumps(_bool_run(1, 1, 1, 1, 1, passed=False))
    )
    summary = summarize(read_runs(tmp_path / "t"))
    assert summary["passed"] == 0


def test_aggregate_exposes_the_spread_for_tweet_verification(tmp_path: Path) -> None:
    out = tmp_path / "out"
    aggregate = write_eval_results(
        _write_tree(tmp_path), out, "hello-proxy", "abc123"
    )
    assert aggregate["usd_per_item_min"] < aggregate["usd_per_item_max"]
    assert aggregate["models"] == 2
    # Every model gets its own flat file so a per-model claim is checkable.
    assert (out / "eval_results.cheap-model.json").exists()
    assert (out / "eval_results.dear-model.json").exists()


def test_run_without_conversations_is_skipped(tmp_path: Path) -> None:
    d = tmp_path / "t" / "1" / "m" / "9"
    d.mkdir(parents=True)
    (d / "r.run.json").write_text(json.dumps({"conversations": []}))
    assert read_runs(tmp_path / "t") == {}


def test_empty_input_refuses_rather_than_writing_zeros(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        summarize({})

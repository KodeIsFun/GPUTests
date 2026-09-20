"""Turn downloaded Benchmarks run output into evidence a tweet may cite.

`kaggle b t download` writes one `.run.json` per (model, item) under
`<out>/<task>/<version>/<model>/<session>/`. Those files nest their metrics
under `conversations[0].metrics`, which `lab.tweets` cannot see — it only reads
top-level keys. This flattens them into `eval_results.json` plus one
`eval_results.<model>.json` per model, so a per-model claim is verifiable
against that model's own file.

Costs arrive in **nanodollars**; everything written here is US dollars.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

NANODOLLARS_PER_USD = 1e9


def _metrics_of(run: dict) -> dict | None:
    conversations = run.get("conversations") or []
    if not conversations:
        return None
    return conversations[0].get("metrics") or {}


def _passed_of(run: dict) -> bool:
    """A Boolean-result task records results[].booleanResult and leaves
    assertions null; an assertion-graded task (W0's regex riddles) records the
    assertion statuses. Support both — and never let an empty list pass."""
    for record in run.get("results") or []:
        if "booleanResult" in record:
            return record.get("booleanResult") is True
    assertions = run.get("assertions") or []
    if not assertions:
        return False
    return all(a.get("status", "").endswith("PASSED") for a in assertions)


def read_runs(download_dir: Path) -> dict[str, list[dict]]:
    """{model_slug: [per-item metric records]} from a downloaded output tree."""
    per_model: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(download_dir.rglob("*.run.json")):
        run = json.loads(path.read_text())
        metrics = _metrics_of(run)
        if metrics is None:
            continue
        # <task>/<version>/<model>/<session>/<file>.run.json
        per_model[path.parent.parent.name].append(
            {
                "input_tokens": metrics.get("inputTokens") or 0,
                "output_tokens": metrics.get("outputTokens") or 0,
                "input_tokens_cost": (
                    int(metrics.get("inputTokensCostNanodollars") or 0)
                    / NANODOLLARS_PER_USD
                ),
                "output_tokens_cost": (
                    int(metrics.get("outputTokensCostNanodollars") or 0)
                    / NANODOLLARS_PER_USD
                ),
                "latency_s": (int(metrics.get("totalBackendLatencyMs") or 0)) / 1000,
                "passed": _passed_of(run),
            }
        )
    return dict(per_model)


def summarize(runs: dict[str, list[dict]]) -> dict:
    """Flat, citable evidence for one model, or for all of them combined."""
    items = [record for records in runs.values() for record in records]
    if not items:
        raise ValueError("no runs with metrics found")
    passed = sum(1 for i in items if i["passed"])
    output_tokens = sum(i["output_tokens"] for i in items)
    return {
        "models": len(runs),
        "items": len(items),
        "passed": passed,
        "score": round(passed / len(items), 4),
        "input_tokens": sum(i["input_tokens"] for i in items),
        "output_tokens": output_tokens,
        # Token counts are what explain a surprising bill, so they must be
        # citable — the per-item average is the number worth quoting.
        "output_tokens_per_item": round(output_tokens / len(items), 4),
        "input_tokens_cost": round(sum(i["input_tokens_cost"] for i in items), 10),
        "output_tokens_cost": round(sum(i["output_tokens_cost"] for i in items), 10),
        "usd_per_item": round(
            sum(
                i["input_tokens_cost"] + i["output_tokens_cost"] for i in items
            )
            / len(items),
            10,
        ),
        "latency_s": round(sum(i["latency_s"] for i in items) / len(items), 4),
    }


def write_eval_results(
    download_dir: Path,
    out_dir: Path,
    task: str,
    git_sha: str | None = None,
) -> dict:
    """Write the aggregate and one per-model file. Returns the aggregate."""
    runs = read_runs(download_dir)
    if not runs:
        raise ValueError(f"no .run.json with metrics under {download_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_model = []
    for model, records in sorted(runs.items()):
        summary = summarize({model: records})
        per_model.append({"model": model, **summary})
        safe = model.replace("/", "_")
        (out_dir / f"eval_results.{safe}.json").write_text(
            json.dumps(
                {"task": task, "model": model, "git_sha": git_sha, **summary},
                indent=2,
            )
            + "\n"
        )

    aggregate = {
        "task": task,
        "git_sha": git_sha,
        **summarize(runs),
        # The spread across models is the story worth telling, and a tweet that
        # compares two models cannot be verified against either model's own
        # file. Surface the extremes — and their ratio — at the top level so
        # such a claim is still traceable to one artifact.
        "usd_per_item_min": min(m["usd_per_item"] for m in per_model),
        "usd_per_item_max": max(m["usd_per_item"] for m in per_model),
        "per_model": per_model,
    }
    cheapest = aggregate["usd_per_item_min"]
    if cheapest > 0:
        aggregate["usd_per_item_max_over_min"] = round(
            aggregate["usd_per_item_max"] / cheapest, 1
        )
    (out_dir / "eval_results.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    return aggregate


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("usage: python -m lab.harvest <download_dir> <out_dir> <task> [git_sha]")
        return 2
    aggregate = write_eval_results(
        Path(argv[1]), Path(argv[2]), argv[3], argv[4] if len(argv) > 4 else None
    )
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))

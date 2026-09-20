from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import re

_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])\$?\d+(?:\.\d+)?(?![A-Za-z0-9])",
)

EVIDENCE_KEYS = frozenset(
    {
        "vram_mb",
        "wall_s",
        "metric",
        "rtf",
        "score",
        "passed",
        "models",
        "items",
        "input_tokens",
        "output_tokens",
        "output_tokens_per_item",
        "input_tokens_cost",
        "output_tokens_cost",
        "usd_per_item",
        "usd_per_item_min",
        "usd_per_item_max",
        "usd_per_item_max_over_min",
        "latency_s",
    }
)


class TweetRejected(ValueError):
    pass


def _normalize(token: str) -> str:
    return token[1:] if token.startswith("$") else token


def _allowed_numbers(evidence: dict[str, Any]) -> list[float]:
    allowed: list[float] = []
    for key in EVIDENCE_KEYS:
        value = evidence.get(key)
        if isinstance(value, bool) or value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, bool) or item is None:
                continue
            if isinstance(item, (int, float)):
                allowed.append(float(item))
    return allowed


def _collect_evidence(payload: dict[str, Any], into: dict[str, Any]) -> None:
    """Index every whitelisted key found at ANY depth, never overwriting.

    Two timings files share one schema (`metric`, `wall_s`...); a plain dict
    merge let the second file clobber the first, which silently made
    cross-lane claims unverifiable. Same-key values accumulate in a list.
    """
    for key, value in payload.items():
        if key in EVIDENCE_KEYS and isinstance(value, (int, float)):
            into.setdefault(key, [])
            if value not in into[key]:
                into[key].append(value)
        elif isinstance(value, dict):
            _collect_evidence(value, into)


def verify_tweet_numbers(tweet: str, evidence: dict[str, Any]) -> None:
    allowed = _allowed_numbers(evidence)
    missing = []
    for raw in _NUMBER.findall(tweet):
        token = float(_normalize(raw))
        if not any(abs(token - item) < 1e-9 for item in allowed):
            missing.append(raw)
    if missing:
        raise TweetRejected(f"numbers not in evidence: {missing}")


def verify_from_files(tweet: str, *paths: Path | None) -> None:
    """Verify against any number of evidence files, merged.

    Takes *paths rather than two fixed slots because a post that compares two
    models cites a number from each model's own eval_results file — no single
    file can witness such a claim. Whitelisted keys are indexed at any depth
    (a timings file's rtf lives at details.whisper.rtf), and same-key values
    from different files accumulate instead of overwriting.
    """
    evidence: dict[str, Any] = {}
    for path in paths:
        if path is None or not path.exists():
            continue
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict):
            _collect_evidence(loaded, evidence)
    verify_tweet_numbers(tweet, evidence)


def compile_tweet(template: str, evidence: dict[str, Any]) -> str:
    tweet = template.format(**evidence)
    if len(tweet) > 280:
        raise TweetRejected(f"tweet is {len(tweet)} chars")
    verify_tweet_numbers(tweet, evidence)
    return tweet

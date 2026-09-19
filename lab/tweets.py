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
        "input_tokens_cost",
        "output_tokens_cost",
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
        if isinstance(value, (int, float)):
            allowed.append(float(value))
    return allowed


def verify_tweet_numbers(tweet: str, evidence: dict[str, Any]) -> None:
    allowed = _allowed_numbers(evidence)
    missing = []
    for raw in _NUMBER.findall(tweet):
        token = float(_normalize(raw))
        if not any(abs(token - item) < 1e-9 for item in allowed):
            missing.append(raw)
    if missing:
        raise TweetRejected(f"numbers not in evidence: {missing}")


def verify_from_files(
    tweet: str,
    timings_path: Path | None = None,
    eval_path: Path | None = None,
) -> None:
    evidence: dict[str, Any] = {}
    for path in (timings_path, eval_path):
        if path is None or not path.exists():
            continue
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict):
            evidence = {**evidence, **loaded}
    verify_tweet_numbers(tweet, evidence)


def compile_tweet(template: str, evidence: dict[str, Any]) -> str:
    tweet = template.format(**evidence)
    if len(tweet) > 280:
        raise TweetRejected(f"tweet is {len(tweet)} chars")
    verify_tweet_numbers(tweet, evidence)
    return tweet

import pytest

from lab.tweets import TweetRejected, compile_tweet, verify_from_files, verify_tweet_numbers


def test_accept_tweet_when_every_number_is_in_evidence() -> None:
    evidence = {"rtf": 0.18, "gpu": "Tesla T4", "wall_s": 42}
    tweet = "Whisper large-v3 on a free Kaggle T4: 0.18× realtime."
    verify_tweet_numbers(tweet, evidence)


def test_reject_invented_dollar_amount() -> None:
    evidence = {"score": 0.8, "model": "gemini-flash"}
    with pytest.raises(TweetRejected):
        verify_tweet_numbers(
            "5 models, $1.40 of Kaggle Benchmarks quota: only 2 passed.",
            evidence,
        )


def test_accept_eval_dollar_when_present() -> None:
    evidence = {
        "models": 5,
        "passed": 2,
        "input_tokens_cost": 1.40,
    }
    tweet = "5 models, $1.40 of Kaggle Benchmarks quota: only 2 passed."
    verify_tweet_numbers(tweet, evidence)


def test_compile_uses_template_and_stays_short() -> None:
    evidence = {"rtf": 0.18, "gpu": "Tesla T4", "kernel": "emdadh/hello-t4"}
    tweet = compile_tweet(
        "Whisper large-v3 on a free {gpu}: {rtf}× realtime. Kernel: {kernel}",
        evidence,
    )
    assert len(tweet) <= 280
    assert "0.18" in tweet


def test_smi_blob_is_not_tweet_evidence() -> None:
    evidence = {
        "gpu": "Tesla T4",
        "smi": "NVIDIA-SMI 15109MiB driver 15.0 CUDA 12.4 $1.40",
        "wall_s": 1.2,
    }
    with pytest.raises(TweetRejected):
        verify_tweet_numbers("$1.40 of quota on driver 15.0", evidence)


def test_cross_lane_metric_from_two_files_survives(tmp_path) -> None:
    """W2: two timings files share the key `metric`; merging must not let the
    second file clobber the first or the comparison post cannot verify."""
    import json

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"metric": 23150.9, "platform": "kaggle"}))
    b.write_text(json.dumps({"metric": 24307.6, "platform": "colab"}))
    verify_from_files("Colab 24307.6 vs Kaggle 23150.9", a, b)


def test_nested_whisper_rtf_is_evidence(tmp_path) -> None:
    import json

    f = tmp_path / "t.json"
    f.write_text(json.dumps({"details": {"whisper": {"rtf": 6.07}}}))
    # "6.07x" would tokenize as 6 (trailing ASCII letter) — use the sign.
    verify_from_files("transcribed 6.07× realtime", f)


def test_nested_smuggled_keys_are_not_evidence(tmp_path) -> None:
    import json

    f = tmp_path / "t.json"
    f.write_text(json.dumps({"details": {"ad": "send $1.40 to driver 15.0"}}))
    with pytest.raises(TweetRejected):
        verify_from_files("$1.40 to driver 15.0", f)

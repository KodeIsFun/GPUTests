import pytest

from lab.tweets import TweetRejected, compile_tweet, verify_tweet_numbers


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

# %%
"""W5 eval sibling "recommend-quant-15gb" — recipes graded against OUR numbers.

Month 2's thesis: hosted models should be able to give advice that survives
contact with a real free GPU. The answer key below is not a textbook value —
every number in MEASURED comes from this lab's own W5 run on a Kaggle T4
(timings.json, committed alongside), which is what makes the grading honest
and the score meaningful.

Deterministic grading only (JSON field checks + regex + numeric windows),
no LLM judge, so the score costs nothing beyond the item prompts.

Run order per AGENTS.md: solo cheap model first, then batch two:

    kaggle b t run recommend-quant-15gb -m gemini-3.5-flash-lite --wait
    kaggle b t run recommend-quant-15gb -m gemini-3.7-flash -m gpt-oss-20b --wait
"""
import json
import re

import kaggle_benchmarks as kbench

# %%
# FILLED POST-HARVEST from projects/2026-W38-gguf-on-t4/results/timings.json
# (W5, Kaggle T4, llama.cpp, Q4_K_M / MXFP4). Sizes in GB as recorded by the
# downloader; tok/s as recorded by the greedy decode timer.
MEASURED = {
    "qwen3_8b_file_gb": None,   # e.g. 5.05
    "qwen3_14b_file_gb": None,  # e.g. 8.99
    "gpt_oss_20b_file_gb": None,
    "qwen3_14b_tg_tok_s": None,     # the reality check for item 5's claim
    "qwen3_8b_tg_tok_s": None,
}
VRAM_MB = 15360
HEADROOM_MB = 1500  # context + CUDA context, per the W5 job's own budgeting


# %%
def extract_json_object(text: str):
    """First balanced {...} in the reply, tolerating markdown fences."""
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        char = cleaned[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", (text or "").replace(",", ""))
    return float(match.group(0)) if match else None


# %%
def grade_recommend(entry: dict, response: str) -> bool:
    """Model+quant for a 7-9B model on 15360 MB: a real quant name, a size
    estimate within a factor of 2 of our measured 8B file, and it must fit."""
    meta = extract_json_object(response)
    if not isinstance(meta, dict):
        return False
    quant = str(meta.get("quant", ""))
    if not re.fullmatch(r"Q[2-8]_[A-Z0-9_]+|IQ[2-4]_[A-Z0-9_]+", quant):
        return False
    size = meta.get("file_gb")
    if not isinstance(size, (int, float)) or size <= 0:
        return False
    if not (MEASURED["qwen3_8b_file_gb"] / 2 <= size <= MEASURED["qwen3_8b_file_gb"] * 2):
        return False
    return size * 1024 + HEADROOM_MB <= VRAM_MB


def grade_14b_fits(entry: dict, response: str) -> bool:
    """Our measured 14B Q4_K_M file leaves >6 GB free on a 15360 MB card."""
    return re.search(r"\byes\b", (response or ""), re.IGNORECASE) is not None


def grade_gpt_oss(entry: dict, response: str) -> bool:
    """The answer key is the measured truth: the MXFP4 file ran on the T4 or
    it did not. FILLED POST-HARVEST."""
    meta = extract_json_object(response)
    if not isinstance(meta, dict):
        return False
    want = MEASURED["gpt_oss_20b_file_gb"] * 1024 + HEADROOM_MB <= VRAM_MB
    return meta.get("fits") is want


def grade_largest_fit(entry: dict, response: str) -> bool:
    """Pick the largest file that fits 15360 MB minus 1.5 GB headroom."""
    got = first_number(response)
    want = None
    for size in (4.9, 8.9, 12.1, 40.6):
        if size * 1024 + HEADROOM_MB <= VRAM_MB:
            want = size
    return got is not None and want is not None and abs(got - want) < 0.05


def grade_speed_claim(entry: dict, response: str) -> bool:
    """A friend claims ~40 tok/s on a 14B Q4 GGUF, llama.cpp, T4. Our measured
    decode speed is the key: 'not plausible' is correct when measured < 20."""
    plausible = MEASURED["qwen3_14b_tg_tok_s"] >= 20
    said_no = re.search(
        r"not\s+(?:plausible|realistic|possible)|no[,.]|unlikely|far\s+(?:slower|off)|"
        r"much\s+slower|too\s+(?:high|optimistic)|closer\s+to",
        (response or ""),
        re.IGNORECASE,
    )
    said_yes = re.search(r"\b(plausible|realistic|reasonable|believable)\b", (response or ""), re.IGNORECASE)
    return (said_yes is not None) if plausible else (said_no is not None)


ITEMS = [
    {
        "prompt": (
            "You have a free Kaggle T4 GPU with exactly 15360 MB of VRAM. "
            "A friend wants to run a 7-9B parameter model locally with "
            "llama.cpp. Reply with a single JSON object and nothing else: "
            '{"quant": "<GGUF quantization name>", "file_gb": <your estimate '
            "of that file's size in GB for a 7-9B model>}"
        ),
        "kind": "recommend",
    },
    {
        "prompt": (
            "A 14B parameter model's Q4_K_M GGUF file is about 9 GB. You have "
            "a GPU with 15360 MB of VRAM and need room for a 4k context. Does "
            "it fit? Answer yes or no with one short sentence."
        ),
        "kind": "fits14b",
    },
    {
        "prompt": (
            "gpt-oss-20b ships as an MXFP4 GGUF file of about 12 GB. Your GPU "
            "has 15360 MB of VRAM. Reply with a single JSON object and nothing "
            'else: {"fits": <true or false>, "tradeoff": "<one sentence about '
            'what you would have to give up, if anything>"}'
        ),
        "kind": "gptoss",
    },
    {
        "prompt": (
            "Which is the largest GGUF file that fits in 15360 MB of VRAM "
            "while leaving 1.5 GB for context? Pick one number: 4.9, 8.9, "
            "12.1, or 40.6 (GB). Reply with only the number."
        ),
        "kind": "largest",
    },
    {
        "prompt": (
            "A friend claims llama.cpp on a Tesla T4 generates about 40 tokens "
            "per second on a 14B Q4_K_M model. In one short sentence: is that "
            "plausible, and what is a realistic number?"
        ),
        "kind": "speed",
    },
]

GRADERS = {
    "recommend": grade_recommend,
    "fits14b": grade_14b_fits,
    "gptoss": grade_gpt_oss,
    "largest": grade_largest_fit,
    "speed": grade_speed_claim,
}


# %%
@kbench.task(name="recommend-quant-15gb")
def recommend_quant_15gb(llm, prompt: str, kind: str) -> bool:
    """One item: prompt the model, grade the reply against the measured key."""
    response = llm.prompt(prompt)
    return GRADERS[kind]({}, response)


# %%
import pandas as pd

# on_failure="continue" so one malformed row still reports — a model whose
# advice would waste someone's afternoon is the finding, not a run to abort.
recommend_quant_15gb.evaluate(
    grid={"llm": [kbench.llm]},
    evaluation_data=pd.DataFrame(ITEMS),
    on_failure="continue",
)

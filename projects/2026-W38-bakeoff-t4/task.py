# %%
"""W2 eval sibling "emit-kernel-metadata" — free-GPU literacy, item 1.

Can a hosted model write a `kernel-metadata.json` that would actually push to
Kaggle as a T4 script kernel? Graded deterministically — JSON field checks and
regex, no LLM judge — so the score costs nothing and every item is a boolean.

Same four models as W0's hello-proxy, so the structured-output score is
comparable against W0's riddle floor:

    kaggle b t run emit-kernel-metadata -m gpt-oss-20b --wait
"""
import json
import re

import kaggle_benchmarks as kbench

# %%
# Expectations are compact strings ("key=value;...") rather than dicts so the
# grader never depends on how the eval harness serializes DataFrame cells.
BASE_SPEC = (
    "Write a Kaggle kernel-metadata.json as a single JSON object and nothing "
    "else. Requirements: a Python script kernel for user emdadh, slug {slug}, "
    "title {title}, GPU enabled, internet enabled, code file job.py."
)

ITEMS = [
    {
        "prompt": BASE_SPEC.format(slug="t4-demo", title="'T4 Demo'"),
        "kind": "metadata",
        "expect": "id=emdadh/t4-demo;title=T4 Demo;kernel_type=script;"
        "language=python;enable_gpu=true;enable_internet=true;code_file=job.py",
    },
    {
        "prompt": BASE_SPEC.format(slug="t4-demo-private", title="'T4 Demo (private)'")
        + " The kernel must be private (not publicly visible).",
        "kind": "metadata",
        "expect": "id=emdadh/t4-demo-private;is_private=true",
    },
    {
        "prompt": "In a Kaggle kernel-metadata.json file, which single key "
        "enables the GPU? Reply with only the exact key name.",
        "kind": "key",
        "expect": "enable_gpu",
    },
    {
        "prompt": "A Kaggle kernel push accepts a session timeout as a whole "
        "number of seconds. A script kernel job needs 1.5 hours. What exact "
        "number of seconds is that? Reply with only the number.",
        "kind": "seconds",
        "expect": "5400",
    },
    {
        "prompt": "A Kaggle kernel push accepts a session timeout as a whole "
        "number of seconds. A script kernel job needs the maximum allowed 8 "
        "hours. What exact number of seconds is that? Reply with only the "
        "number.",
        "kind": "seconds",
        "expect": "28800",
    },
]


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


def parse_expect(encoded: str) -> list[tuple[str, str]]:
    pairs = []
    for part in encoded.split(";"):
        key, _, value = part.partition("=")
        if key and value:
            pairs.append((key.strip(), value.strip()))
    return pairs


def grade(kind: str, expect: str, response: str) -> bool:
    if kind == "metadata":
        meta = extract_json_object(response)
        if not isinstance(meta, dict):
            return False
        for key, want in parse_expect(expect):
            if key not in meta:
                return False
            got = meta[key]
            if want.lower() == "true":
                # Kaggle's schema wants real JSON booleans — the string "true"
                # would not push, so it fails here too.
                if got is not True:
                    return False
            elif want.lower() == "false":
                if got is not False:
                    return False
            elif str(got).strip().lower() != want.lower():
                return False
        return True
    if kind == "key":
        return re.search(rf"\b{re.escape(expect)}\b", response or "") is not None
    if kind == "seconds":
        flat = (response or "").replace(",", "")
        # (?!\d)(?!\.\d) rejects "54000" and "5400.5" while still passing a
        # sentence that ends in "28800."
        return re.search(rf"(?<![\d.]){expect}(?!\d)(?!\.\d)", flat) is not None
    return False


# %%
@kbench.task(name="emit-kernel-metadata")
def emit_kernel_metadata(llm, prompt: str, kind: str, expect: str) -> bool:
    """One item: prompt the model, grade the reply deterministically."""
    response = llm.prompt(prompt)
    return grade(kind, expect, response)


# %%
import pandas as pd

# on_failure="continue" so one refused row still reports — a model that cannot
# emit the schema is the finding, not a run to abort.
emit_kernel_metadata.evaluate(
    grid={"llm": [kbench.llm]},
    evaluation_data=pd.DataFrame(ITEMS),
    on_failure="continue",
)

# %%
"""W0/1 "Hello Proxy" — calibrate what one Model Proxy call actually costs.

Five riddles, one prompt each, graded by a deterministic regex so the score is
not itself an LLM expense. The CLI's -m flag binds kbench.llm (it sets
LLM_DEFAULT), so this single file is reused for every model and the resulting
$/call numbers stay comparable across models:

    kaggle b t run hello-proxy -m gemini-3.5-flash-lite -m gpt-5.4-nano --wait

Budget for the W0 sweep is $2 total. See PLAN.md "Cost shape".
"""
import re

import kaggle_benchmarks as kbench

# %%
# One unambiguous word per answer — an LLM judge would double the bill.
RIDDLES = [
    {
        "riddle": "I have cities but no houses, forests but no trees, "
        "and water but no fish. What am I? Answer with one word.",
        "answer": "map",
    },
    {
        "riddle": "What has keys but no locks, space but no room, "
        "and you can enter but not go in? Answer with one word.",
        "answer": "keyboard",
    },
    {
        "riddle": "What gets wetter the more it dries? Answer with one word.",
        "answer": "towel",
    },
    {
        "riddle": "What has a face and two hands but no arms or legs? "
        "Answer with one word.",
        "answer": "clock",
    },
    {
        "riddle": "What has many teeth but cannot bite? Answer with one word.",
        "answer": "comb",
    },
]

# %%
@kbench.task(name="hello-proxy")
def hello_proxy(llm, riddle: str, answer: str) -> bool:
    """Ask one riddle; True when the reply contains the expected word."""
    response = llm.prompt(riddle)
    verdict = kbench.assertions.assert_contains_regex(
        rf"\b{re.escape(answer)}\b",
        response,
        expectation=f"reply names '{answer}'",
        flags=re.IGNORECASE,
    )
    return bool(verdict.passed)


# %%
import pandas as pd

# on_failure="continue" so a single bad row still reports its cost — W0 is a
# calibration run, and a refused answer is itself a data point.
hello_proxy.evaluate(
    grid={"llm": [kbench.llm]},
    evaluation_data=pd.DataFrame(RIDDLES),
    on_failure="continue",
)

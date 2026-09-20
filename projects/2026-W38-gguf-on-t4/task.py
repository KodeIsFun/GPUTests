# %%
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="hello-proxy")
def hello_proxy(llm):
    """Tiny W0 calibration item. Replace before any real spend."""
    response = llm.prompt("Reply with the single digit 4 and nothing else.")
    kbench.assertions.assert_in("4", response, expectation="model returns 4")


hello_proxy.run(kbench.llm)

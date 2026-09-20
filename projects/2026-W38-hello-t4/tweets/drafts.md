# W0/1 tweet drafts — rewritten 2026-09-19

First pass was spec-sheet posts ("15360 MB VRAM, 21162.1 GFLOP/s fp16"). True,
verified, and useless — no reader has a reason to care.

Rewritten to lead with friction and surprise. Measurements are evidence, not
headline. Every draft below is **verified** against the real artifacts:
`verify_from_files(tweet, *paths)`.

## Priority order

1. Gotchas (C, D, E) — highest value, no number needed
2. Surprises (A, B) — the number proves the surprise

---

## A. The cost surprise        [eval_results.json]  216 chars

Budgeted a couple of dollars to find out what LLM evals cost on Kaggle's free tier.

Four models, same twenty questions.

Cheapest: $0.0000069 a question.
Dearest: $0.0004293.

I was off by three orders of magnitude.

---

## B. Why it happened          [eval_results.json + both per-model files]  226 chars

Two models. Same one-word answer.

One spent $0.0000069. The other spent $0.0004293, because it burned 110.4 output
tokens thinking before it answered.

You're not paying for your prompt. You're paying for its inner monologue.

---

## C. The model list lies      [no numbers]  216 chars

Kaggle's Benchmarks model list is a catalogue, not a promise.

It will list a model as available, then refuse to schedule it. The error names
every model you passed, not the broken one.

Add new models one at a time.

---

## D. Pushing isn't free       [no numbers]  205 chars

Pushing a Kaggle Benchmarks task is not free.

Creating the task runs it once against a default model, before you've chosen one.

My first upload quietly spent quota. Budget for the push, not just the run.

---

## E. The orphan VM            [no numbers]  216 chars

Found one of my Colab VMs still running long after the script had finished,
quietly burning compute units.

`colab new` doesn't clean up after itself. `colab run` does.

If you automate Colab, go hunting for orphans.


---

## F. The rule that rejected my own drafts   [no measured digits]  653 chars

Process post. Breaks down the issue/solution arc rather than reporting a
measurement.

New rule in my free-GPU lab: every digit I publish has to exist in a results file. A script rejects the post if it doesn't.

Then it rejected three of my own drafts.

None of them were wrong, and that turned out to be the interesting part. A check like this can't tell a lie from a fact you computed but never wrote down. Some of the numbers were buried in a nested field. The rest were token counts I'd never saved in a quotable shape.

So the fix wasn't better writing. It was saving the evidence differently — one results file per model, with the numbers I actually want to quote up top.

Still annoying. Still the only reason I trust my own numbers.

**Honesty note:** this verifies *vacuously* — the gate only inspects digits and
this post spells its counts out in words, so nothing here is actually vouched
for by a results file. The gate guarantees no invented *measurements*; it says
nothing about process claims. Do not read a PASS here as evidence that the
counts are right.

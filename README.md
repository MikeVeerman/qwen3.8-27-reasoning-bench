# Qwen3.8-27B: does `reasoning_effort=medium` beat llama.cpp's `--reasoning-budget` cutoff?

Benchmark of **Qwen3.8-27B-UD-Q6_K_XL** on llama-swap / llama.cpp (build `b9867`,
`152d337fa`) running on a Framework Desktop (Strix Halo, 122 GB unified memory).

## The claim under test

The production config uses a hard reasoning cutoff, because the model sometimes
reasons for 90+ minutes:

```
--reasoning-budget 8192
--reasoning-budget-message "Time to stop thinking. Give the final answer or make the tool call now."
```

The claim (from Reddit) is that this hard cutoff is a **brute-force solution**
that produces worse results than simply setting
`--chat-template-kwargs '{"reasoning_effort": "medium"}'` and letting the model
stop on its own.

**Verdict so far: not supported on this model.** Accuracy is identical; `medium`
costs 39% more tokens and is the only setting that fails to answer at all.

---

## 1. What `reasoning_effort` actually does

Extracted from the chat template embedded in the GGUF. This is the mechanism,
and it is not what the claim assumes:

| value | what the template injects | prompt tokens ("Say OK") |
|---|---|---|
| `xhigh` (**default**) | *"Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."* | 54 |
| `medium` | **nothing at all** — no reasoning instruction | 12 |
| `low` | *"Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."* | 42 |
| `high` | silently aliased to `xhigh` | 54 |

`medium` is **not** a middle setting. It is the *neutral* prompt with the
verbosity nudge removed. Critically, the `xhigh` text ends with *"...clarity in
the final answer"*, a clause that actively keeps output tight. Removing it makes
the model **more** verbose, not less — which is the opposite of the assumption
behind the claim, and it is the single most reproducible finding here.

Only `xhigh`, `medium`, `low` are accepted; anything else raises a template
exception.

## 2. Both knobs are per-request

`thinking_budget_tokens` in the request body overrides the server's
`--reasoning-budget` (`tools/server/server-common.cpp:1119`), falling back to the
server flag only when absent or `-1`. Combined with per-request
`chat_template_kwargs`, every arm runs against **one loaded model** — identical
weights, sampler, and KV cache. No config edits, no restarts, no confound.

Verified: `thinking_budget_tokens: 300` truncates thinking at ~300 tokens and
injects the configured stop message.

---

## 3. Results — rounds 1 & 2

### Round 1 (100 responses): effort levels, cutoff inert

25 graded tasks x 3 efforts x 3 seeds, on easy/medium tasks.

```
effort     acc%  compl.tok  think.chars   sec
xhigh     100.0        247          533  13.9     <- default
medium    100.0        522          788  24.8
low        93.9        362          474  17.4
```

Paired on identical task+seed: **33 ties, 0 wins either way**, while `medium`
cost **+111% tokens** and **+79% latency**. `low` was the only setting that
actually saved tokens, and it lost accuracy.

**This round could not test the real claim** — the 8192 cutoff fired **0 times
in 99 responses**, so all three arms ran with the cutoff inert. Kept here
because the verbosity ordering it establishes is real and replicated.

### Round 2 (28 responses): the actual head-to-head, at production scale

7 hard combinatorics tasks x 2 seeds, both factors varied per request:

- **A** = `xhigh` + 8192 cutoff (status quo, the "brute force" solution)
- **B** = `medium` + unlimited (the Reddit proposal)

```
                        A: xhigh+8192    B: medium+unlimited
accuracy                      12/14                  12/14
total completion tokens      74,373                103,170   (+39%)
mean tokens/task              5,312                  7,369
no answer produced                0                      2
hit 16,384 ceiling                0                      2

Paired: A-only-correct=0   B-only-correct=0   both=12   neither=2
```

**Zero discordant pairs.** In 14 paired comparisons the two configs never
disagreed on a single answer. They differed only in cost.

### The subset that actually tests the claim

The cutoff fired on only **4 of 14** runs. Everywhere else it is inert and the
comparison is vacuous — this is the main limitation of rounds 1-2.

| task | seed | A (cutoff fired) | B (medium, unlimited) |
|---|---|---|---|
| h12_nozero | 22 | **correct**, 8,584 tok | correct, 6,586 tok |
| h2_perms | 11 | WRONG (8), 8,215 tok | WRONG (no answer), 16,384 tok |
| h2_perms | 22 | WRONG (120), 8,217 tok | WRONG (no answer), 16,384 tok |
| h3_subsets | 11 | **correct**, 8,218 tok | correct, 12,216 tok |

```
accuracy when the cutoff fired:   A = 2/4     B = 2/4
cases where the cutoff LOST an answer medium got right:   0
```

Two things stand out:

1. **Truncation did not destroy answers.** On `h3_subsets` and `h12_nozero` the
   guillotine cut reasoning mid-thought and the model *still* produced the
   correct answer.
2. **`medium` reproduces the runaway failure the cutoff exists to prevent.** On
   `h2_perms`, in **both** seeds, `medium` burned the entire 16,384-token ceiling
   and returned **no answer at all** — 2x the cost for nothing. The cutoff arm
   terminated and produced a (wrong) answer. A wrong answer and a non-answer are
   different failure modes, and only one of them wastes the whole budget.

### Runaway reasoning is real

Separately, `xhigh` + unlimited budget on "ordered pairs (a,b), a+b=1000, no zero
digit" (answer 738) produced **20,000 completion tokens / 55,964 characters of
reasoning**, hit the ceiling, and never emitted an answer — still mid-sentence.
At ~15 tok/s that is ~22 minutes for nothing. The problem motivating the cutoff
is genuine. (Single sample, and that prompt asked for verification by multiple
methods, so treat it as corroboration rather than a rate.)

---

## 4. Round 3 — adversarial (in progress)

Rounds 1-2 have a real weakness: the cutoff fired on only 4 of 14 runs. Round 3
is built to **force the guillotine** and to **stack the deck for `medium`**, so
the claim gets its best shot.

**Tasks chosen to favour `medium`.** Category A is long *mechanical* execution
(Collatz from 27 = 111 steps; a 30-step modular recurrence; summing d(n) over
100 integers). There are no alternatives to weigh and no assumptions to
validate, so `xhigh`'s instruction is pure overhead while `medium` just executes.
And these have **no partial credit** — the answer exists only at the end of the
chain, so truncating at step 70 of 111 destroys it. Maximum guillotine damage
where `xhigh` is maximally wasteful. Category B forces the cutoff by brute size
(counts in the millions).

**A uniform `SHOW_WORK` instruction** is appended identically to every arm on
category A, purely to push both arms past the 8192 boundary so the cutoff is
under test rather than inert.

**Three arms, so the cutoff is finally isolated:**

| arm | effort | budget | |
|---|---|---|---|
| A | xhigh | 8192 | status quo |
| B | medium | unlimited | the Reddit proposal |
| D | medium | 8192 | medium **with** the cutoff |

`B vs D` isolates the guillotine at constant effort — the direct test of whether
truncation itself hurts, which **no result in rounds 1-2 can answer**, since A
and B differ in both factors. `A vs D` isolates the effort prompt at constant
budget.

---

## 5. Honest limitations

- **Small n on the decisive subset.** 4 of 14 runs. Round 3 targets exactly this.
- **A vs B is confounded by design.** The two arms differ in effort *and* budget,
  so rounds 1-2 cannot attribute a difference to the cutoff alone. Correct for a
  deployment choice, wrong for isolating a mechanism. Arm D fixes it.
- **One model, one quant, one task family.** Combinatorics with integer answers.
  Nothing here transfers automatically to coding or agentic tool use.
- **Throughput is bandwidth-bound.** ~15 tok/s aggregate regardless of
  concurrency, so wall-clock latency under 4-way batching is inflated. Token
  counts are the load-independent cost metric and are used throughout.
- **`medium` was never given the cutoff in rounds 1-2**, which is arguably its
  best configuration. Arm D tests it.

## 6. Reproducing

```bash
python3 ground_truths.py          # re-verify all 18 answers from scratch (18/18)
python3 scripts/run3h.py          # rounds 1-2 head-to-head
SEED=11 python3 scripts/round3.py # adversarial round
python3 scripts/final.py          # analysis
```

Every benchmark answer is brute-forced in `ground_truths.py` rather than taken
from memory or an external source. A logic puzzle was cut from the task set
during development because verification showed it admitted 8 solutions, not the
unique one intended.

Raw per-response records (including full reasoning traces, token counts,
guillotine flags and finish reasons) are in `data/*.jsonl`.

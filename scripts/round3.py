#!/usr/bin/env python3
"""
Round 3 -- adversarial. Built to give the Reddit claim its BEST possible shot.

Rounds 1-2 found no case where the 8192 cutoff lost an answer medium got
right. But that was on tasks where the cutoff usually did not even fire (2 of
7). So round 3 does two things: force the guillotine, and stack the deck for
medium. If the cutoff still wins HERE, the conclusion is strong. If it loses,
the Reddit claim was right and I want to find that out.

WHY THESE TASKS FAVOUR MEDIUM
-----------------------------
The xhigh template text is "...validate key assumptions, consider plausible
alternatives, and prioritize correctness, consistency, and clarity...".

Category A is long MECHANICAL execution: iterate a recurrence 30 times, sum
d(n) over 100 integers, run Collatz 111 steps. There are no plausible
alternatives to weigh and no assumptions to validate -- the task is pure grind.
So xhigh's instruction is pure overhead, spending budget on meta-reasoning
while medium executes directly. That is the theoretical case for medium.

Simultaneously these tasks have NO PARTIAL CREDIT: the answer exists only at
the end of the chain. Truncating at step 70 of 111 yields nothing. So the
guillotine is maximally destructive exactly where xhigh is maximally wasteful.
That is the worst case for the cutoff, by construction.

Category B forces the guillotine by brute size -- counts in the millions that
cannot be enumerated by hand inside 8192 tokens.

FORCING THE CUTOFF TO FIRE
--------------------------
Category A tasks carry a uniform SHOW_WORK instruction that inflates reasoning
length. It is appended identically for every arm, so it cannot bias the
comparison -- it only pushes both arms past the 8192 boundary so the cutoff is
actually tested rather than sitting inert.

ARMS -- three, so the cutoff can finally be isolated
----------------------------------------------------
  A_cutoff       xhigh  + 8192       status quo
  B_medium_free  medium + unlimited  the Reddit proposal
  D_medium_cut   medium + 8192       medium WITH the cutoff

  B vs D -> isolates the guillotine at constant effort (does truncation hurt?)
  A vs D -> isolates the effort prompt at constant budget
  A vs B -> the deployment choice

KNOWN RISK: if these tasks are too hard, everything fails in every arm and the
round is uninformative again. Watch the first two tasks; if both arms score
zero, swap in easier ones.
"""
import json
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import phase2_budget_vs_effort as P

P.MAX_TOKENS = 16384
OUT = pathlib.Path(__file__).resolve().parent / "results-round3.jsonl"
SEED = int(os.environ.get("SEED", "11"))

ARMS = [
    ("A_cutoff",      "xhigh",  8192),
    ("B_medium_free", "medium", P.UNLIMITED),
    ("D_medium_cut",  "medium", 8192),
]

SHOW_WORK = (" Show every intermediate value explicitly as you go, and verify "
             "your result by a second independent method before answering.")

# (id, answer, prompt) -- every answer brute-forced in Python beforehand.
# Category A = long mechanical grind, no partial credit, favours medium.
# Category B = brute-size enumeration, forces the guillotine by difficulty.
TASKS = [
    ("a3_collatz", 111,
     "Starting from 27, apply the Collatz rule (n -> n/2 if n is even, "
     "n -> 3n+1 if n is odd) repeatedly until reaching 1. How many steps does "
     "it take?" + SHOW_WORK),
    ("a1_recurrence", 594,
     "Define a_1 = 3 and a_{n+1} = (a_n^2 + 5) mod 1009. What is a_30?"
     + SHOW_WORK),
    ("a2_divisorsum", 482,
     "Let d(n) be the number of positive divisors of n. Compute "
     "d(1) + d(2) + ... + d(100)." + SHOW_WORK),
    ("a4_nonconsec", 982,
     "For each k, let g(k) be the number of subsets of {1,2,...,k} containing "
     "no two consecutive integers (the empty set counts). Compute "
     "g(2) + g(3) + ... + g(12)." + SHOW_WORK),
    ("b2_perms10", 101042,
     "How many permutations p of (1,2,...,10) contain no three consecutive "
     "positions i, i+1, i+2 that are strictly increasing and no three "
     "consecutive positions that are strictly decreasing?"),
    ("b1_matrices77", 3110940,
     "How many 7x7 matrices with entries in {0,1} have exactly two 1s in every "
     "row and exactly two 1s in every column?"),
    ("b3_graphs7", 1866256,
     "How many labeled simple graphs on the vertex set {1,2,...,7} are "
     "connected?"),
    ("h2_perms", 1695,
     "How many permutations p of (1,2,...,8) satisfy both: p(i) != i for every "
     "i, and |p(i) - p(i+1)| != 1 for every i from 1 to 7?"),
]

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add((r["task"], r["seed"], r["arm"]))
        except Exception:
            pass
    print("resuming; %d results already present" % len(done), flush=True)

print("ROUND 3 (adversarial): %d tasks x %d arms, seed=%d, max_tokens=%d"
      % (len(TASKS), len(ARMS), SEED, P.MAX_TOKENS), flush=True)

t0 = time.time()
spent = 0

with OUT.open("a") as out:
    for ti, task in enumerate(TASKS, 1):
        todo = [a for a in ARMS if (task[0], SEED, a[0]) not in done]
        if not todo:
            continue
        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            recs = list(ex.map(lambda a: P.call(task, SEED, a), todo))
        for r in sorted(recs, key=lambda x: x["arm"]):
            out.write(json.dumps(r) + "\n")
            spent += r.get("completion_tokens") or 0
        out.flush()
        print("\n[task %d/%d] %s  want=%s   (%.0fm elapsed, %d tok spent)"
              % (ti, len(TASKS), task[0], task[1],
                 (time.time() - t0) / 60, spent), flush=True)
        for r in sorted(recs, key=lambda x: x["arm"]):
            if r.get("error"):
                print("    %-14s ERROR %s" % (r["arm"], r["error"][:70]), flush=True)
            else:
                print("    %-14s %s got=%10s tok=%-6d think_ch=%-7d "
                      "guillo=%-5s maxtok=%-5s %6.0fs"
                      % (r["arm"], "OK " if r["correct"] else "XX ",
                         str(r["got"]), r["completion_tokens"],
                         r["reasoning_chars"], str(r["guillotine"]),
                         str(r["hit_max_tokens"]), r["elapsed_s"]), flush=True)
print("\nDONE", flush=True)

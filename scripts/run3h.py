#!/usr/bin/env python3
"""
The real comparison, at the production 8192 scale, sized to a ~3h token budget.

Hardware reality: this box does ~15 tok/s aggregate regardless of concurrency
(memory-bandwidth bound), so 3h == ~165k completion tokens. Spending them:

  * 2 arms, not 4. A vs B is the deployment decision actually being made.
    Arm C is unnecessary because whether the cutoff bit is read directly off
    arm A guillotine flag -- the smoke test taught me that. Arm D is a
    nice-to-have. Both can be added later; this script resumes.

  * 7 tasks chosen because the smoke test measured natural xhigh thinking at
    335-1960 tokens for h5/h11/h10/h8 -- all far below 8192, so the cutoff
    would NEVER bite on them and they would be pure wasted tokens here.
    These 7 are the untested hard ones, plus the a+b=1000 problem that blew
    past 20k thinking tokens in the earlier probe.

  * 2 workers, paired by task, so A and B of the same task run together under
    identical load and each task yields a COMPLETE paired result every ~25min.
    Total wall time is bandwidth-bound either way, so fewer workers just means
    usable results arrive sooner and I can stop early.

Caveat recorded up front: A and B differ in BOTH effort and budget, so a
difference cannot be attributed to the cutoff alone. That is the right
comparison for a deployment choice, not for isolating a mechanism.
"""
import json, pathlib, threading, time
from concurrent.futures import ThreadPoolExecutor
import phase2_budget_vs_effort as P

P.MAX_TOKENS = 16384
OUT = pathlib.Path(__file__).resolve().parent / "results-phase2.jsonl"
import os
SEED = int(os.environ.get("SEED", "11"))

ARMS = [
    ("A_cutoff",      "xhigh",  8192),
    ("B_medium_free", "medium", P.UNLIMITED),
]

WANT = ["h1_strings", "h2_perms", "h3_subsets", "h4_matrices", "h6_trees", "h9_cube"]
EXTRA = [("h12_nozero", 738,
          "Find the number of ordered pairs of positive integers (a, b) with "
          "a + b = 1000 such that neither a nor b has a zero digit.")]
TASKS = [t for t in P.TASKS if t[0] in WANT] + EXTRA

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add((r["task"], r["seed"], r["arm"]))
        except Exception:
            pass
    print("resuming; %d results already present" % len(done), flush=True)

print("%d tasks x %d arms, seed=%d, max_tokens=%d"
      % (len(TASKS), len(ARMS), SEED, P.MAX_TOKENS), flush=True)

t0 = time.time()
spent = 0

with OUT.open("a") as out:
    for ti, task in enumerate(TASKS, 1):
        todo = [a for a in ARMS if (task[0], SEED, a[0]) not in done]
        if not todo:
            continue
        with ThreadPoolExecutor(max_workers=2) as ex:
            recs = list(ex.map(lambda a: P.call(task, SEED, a), todo))
        for r in sorted(recs, key=lambda x: x["arm"]):
            out.write(json.dumps(r) + "\n")
            spent += r.get("completion_tokens") or 0
        out.flush()
        el = (time.time() - t0) / 60
        print("\n[task %d/%d] %s  want=%s   (%.0fm elapsed, %d tok spent)"
              % (ti, len(TASKS), task[0], task[1], el, spent), flush=True)
        for r in sorted(recs, key=lambda x: x["arm"]):
            if r.get("error"):
                print("    %-14s ERROR %s" % (r["arm"], r["error"][:70]), flush=True)
            else:
                print("    %-14s %s got=%8s  tok=%-6d think_ch=%-7d guillo=%-5s maxtok=%-5s %6.0fs"
                      % (r["arm"], "OK " if r["correct"] else "XX ", str(r["got"]),
                         r["completion_tokens"], r["reasoning_chars"],
                         str(r["guillotine"]), str(r["hit_max_tokens"]),
                         r["elapsed_s"]), flush=True)
print("\nDONE", flush=True)

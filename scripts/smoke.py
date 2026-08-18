#!/usr/bin/env python3
"""
Fast smoke test of the SAME mechanism as phase 2, scaled down ~8x.

The real question is whether a hard truncation cutoff is worse than letting
reasoning_effort=medium stop on its own. That mechanism does not depend on the
cutoff being 8192 specifically -- it depends on the cutoff landing BELOW what
the model naturally wants to think. So: cutoff 1024 instead of 8192, and tasks
in the matching difficulty band. ~8x cheaper, same comparison.

  A_cut1024   xhigh  + 1024 cutoff   <- "brute force", scaled down
  B_med_free  medium + unlimited     <- the Reddit proposal
  C_xhigh_ref xhigh  + unlimited     <- what xhigh naturally wants

Arm C is what makes this self-diagnosing: if C's thinking exceeds 1024 tokens
for a task, then the cutoff genuinely bit in arm A and that task is
informative. If it didn't, the task was too easy and I say so rather than
reporting a vacuous tie.
"""
import json
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import phase2_budget_vs_effort as P

P.MAX_TOKENS = 3072
OUT = pathlib.Path(__file__).resolve().parent / "results-smoke.jsonl"

CUT = 1024
ARMS = [
    ("A_cut1024",   "xhigh",  CUT),
    ("B_med_free",  "medium", P.UNLIMITED),
    ("C_xhigh_ref", "xhigh",  P.UNLIMITED),
]

WANT = {"h5_factorial", "h10_digits", "h11_balls", "h8_dominoes"}
TASKS = [t for t in P.TASKS if t[0] in WANT]
SEED = 11

jobs = [(t, SEED, a) for t in TASKS for a in ARMS]
print(f"{len(TASKS)} tasks x {len(ARMS)} arms = {len(jobs)} requests "
      f"(cutoff={CUT}, max_tokens={P.MAX_TOKENS})", flush=True)

t0 = time.time()
lock = threading.Lock()
n = 0

with OUT.open("w") as out:
    def work(job):
        global n
        r = P.call(*job)
        with lock:
            n += 1
            out.write(json.dumps(r) + "\n")
            out.flush()
            if r.get("error"):
                print(f"[{n}/{len(jobs)}] {r['task']:14s} {r['arm']:12s} "
                      f"ERROR {r['error'][:60]}", flush=True)
            else:
                print(f"[{n}/{len(jobs)}] {r['task']:14s} {r['arm']:12s} "
                      f"{'OK ' if r['correct'] else 'XX '} "
                      f"got={str(r['got']):>7s} want={r['expected']:<6d} "
                      f"tok={r['completion_tokens']:<5d} "
                      f"think_ch={r['reasoning_chars']:<6d} "
                      f"guillo={str(r['guillotine']):5s} "
                      f"maxtok={str(r['hit_max_tokens']):5s} "
                      f"{r['elapsed_s']:5.0f}s ({(time.time()-t0)/60:.0f}m)",
                      flush=True)
        return r

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(work, jobs))

print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)

# ---- summary -------------------------------------------------------------
ok = [r for r in res if not r.get("error")]
nat = {r["task"]: r["reasoning_chars"] for r in ok if r["arm"] == "C_xhigh_ref"}
CH_PER_TOK = 3.6  # rough chars-per-token for this model's reasoning text

print("\nDid the cutoff actually bite? (arm C = natural xhigh thinking length)")
informative = set()
for t in sorted(nat):
    est_tok = nat[t] / CH_PER_TOK
    bit = est_tok > CUT
    if bit:
        informative.add(t)
    print(f"  {t:14s} natural think ~{est_tok:6.0f} tok  -> cutoff {CUT} "
          f"{'BIT' if bit else 'did NOT bite (task too easy)'}")

print(f"\nPer-arm results ({len(informative)} of {len(nat)} tasks informative)")
print(f"{'arm':12s} {'acc':>7s} {'acc(informative)':>18s} {'mean tok':>9s} "
      f"{'mean think_ch':>14s} {'guillotined':>12s}")
for name, _, _ in ARMS:
    rs = [r for r in ok if r["arm"] == name]
    if not rs:
        continue
    inf = [r for r in rs if r["task"] in informative]
    acc = 100 * sum(r["correct"] for r in rs) / len(rs)
    acci = (100 * sum(r["correct"] for r in inf) / len(inf)) if inf else float("nan")
    print(f"{name:12s} {acc:6.0f}% {acci:17.0f}% "
          f"{sum(r['completion_tokens'] for r in rs)/len(rs):9.0f} "
          f"{sum(r['reasoning_chars'] for r in rs)/len(rs):14.0f} "
          f"{sum(r['guillotine'] for r in rs):12d}")

print("\nPer-task detail (correct? / tokens)")
print(f"{'task':14s}" + "".join(f"{a[0]:>16s}" for a in ARMS))
for t, exp, _ in TASKS:
    line = f"{t:14s}"
    for name, _, _ in ARMS:
        r = next((x for x in ok if x["task"] == t and x["arm"] == name), None)
        line += f"{('OK' if r['correct'] else 'XX')+' '+str(r['completion_tokens']):>16s}" if r else f"{'-':>16s}"
    print(line + f"   (want {exp})")

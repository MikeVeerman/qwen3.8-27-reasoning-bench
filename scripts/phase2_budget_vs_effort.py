#!/usr/bin/env python3
"""
Phase 2 — the actual question:

  Is llama.cpp's hard `--reasoning-budget 8192` cutoff ("brute force") a WORSE
  way to stop Qwen3.8-27B over-thinking than simply setting
  reasoning_effort=medium and letting the model stop on its own?

2x2 factorial. Both factors are set PER REQUEST, so all four arms hit one
loaded model with identical weights and sampler:

  factor 1  reasoning_effort   : xhigh (template default) | medium
  factor 2  thinking_budget_tokens : 8192 (hard cutoff)   | unlimited

  A_cutoff       xhigh  + 8192       <- status quo, the "brute force" solution
  B_medium_free  medium + unlimited  <- the Reddit proposal
  C_xhigh_free   xhigh  + unlimited  <- unconstrained ceiling (the 90-min problem)
  D_medium_cut   medium + 8192       <- does medium still need a cutoff?

All jobs go through a 4-worker queue into the server's 4 slots, because the run
is throughput-bound. That means wall-clock latency per request is inflated by
batching, so the PRIMARY cost metric here is token counts (load-independent);
single-stream latency is measured separately afterwards.

Tasks are hard combinatorics/number-theory problems with single integer
answers, every ground truth brute-forced in Python beforehand. They are chosen
to be hard enough that xhigh genuinely wants to think for more than 8192
tokens -- otherwise the cutoff never binds and the experiment is vacuous.
Whether that held is checked at analysis time via the `guillotine` field.
"""
import json
import pathlib
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-27b-q6-mtp-250k"
OUT = pathlib.Path(__file__).resolve().parent / "results-phase2.jsonl"

UNLIMITED = 1_000_000
MAX_TOKENS = 16384          # uniform across arms for fairness
SEEDS = [int(s) for s in __import__("os").environ.get("SEEDS", "11").split(",")]

ARMS = [
    ("A_cutoff",      "xhigh",  8192),
    ("B_medium_free", "medium", UNLIMITED),
    ("C_xhigh_free",  "xhigh",  UNLIMITED),
    ("D_medium_cut",  "medium", 8192),
]

SUFFIX = ("\n\nGive your final answer as a single integer on the last line, in "
          "exactly this form:\nANSWER: <integer>")

# ground truths all verified by brute force in Python
TASKS = [
    ("h1_strings", 685,
     "How many strings of length 10 over the alphabet {1,2,3} have no two "
     "adjacent characters equal AND do not contain '123' as a contiguous "
     "substring?"),
    ("h2_perms", 1695,
     "How many permutations p of (1,2,...,8) satisfy both: p(i) != i for every "
     "i, and |p(i) - p(i+1)| != 1 for every i from 1 to 7?"),
    ("h3_subsets", 3544,
     "How many nonempty subsets of {1,2,...,20} contain no two consecutive "
     "integers and have an element sum divisible by 5?"),
    ("h4_matrices", 2040,
     "How many 5x5 matrices with entries in {0,1} have exactly two 1s in every "
     "row and exactly two 1s in every column?"),
    ("h5_factorial", 33,
     "What is the smallest positive integer n such that n! is divisible by "
     "2^30 * 3^15 * 5^7?"),
    ("h6_trees", 36015,
     "How many labeled trees on the vertex set {1,2,...,8} have vertex 1 with "
     "degree exactly 3?"),
    ("h8_dominoes", 281,
     "In how many ways can a 4x6 rectangle be tiled by 1x2 dominoes?"),
    ("h9_cube", 333,
     "How many ways are there to colour the 8 vertices of a cube using 3 "
     "available colours, where two colourings are considered the same if one "
     "can be rotated into the other? Reflections are NOT allowed. Colours need "
     "not all be used."),
    ("h10_digits", 81,
     "How many 4-digit positive integers have digit sum exactly 20 and are "
     "divisible by 11?"),
    ("h11_balls", 68,
     "In how many ways can 10 identical balls be distributed into 4 distinct "
     "boxes so that no box contains more than 4 balls?"),
]


def graded(content, expected):
    hits = re.findall(r"ANSWER\s*:\s*(.+)", content or "", re.IGNORECASE)
    if not hits:
        return False, None
    s = hits[-1]
    s = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("**", "").replace("`", "").replace("\\", "")
    s = re.sub(r"[,\s]", "", s)
    m = re.search(r"-?\d+", s)
    if not m:
        return False, None
    v = int(m.group())
    return v == expected, v


def call(task, seed, arm):
    name, effort, budget = arm
    tid, expected, prompt = task
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + SUFFIX}],
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "thinking_budget_tokens": budget,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=5400) as r:
            d = json.loads(r.read())
        err = None
    except Exception as exc:
        d, err = None, f"{type(exc).__name__}: {exc}"
    el = time.time() - t0

    rec = {"task": tid, "expected": expected, "seed": seed, "arm": name,
           "effort": effort, "budget": budget,
           "elapsed_s": round(el, 2), "error": err}
    if d:
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        rc = msg.get("reasoning_content") or ""
        usage = d.get("usage", {})
        tim = d.get("timings", {})
        ok, got = graded(content, expected)
        rec.update({
            "correct": ok, "got": got,
            "finish_reason": d["choices"][0]["finish_reason"],
            "hit_max_tokens": d["choices"][0]["finish_reason"] == "length",
            "guillotine": "Time to stop thinking" in rc,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_chars": len(rc),
            "answer_chars": len(content),
            "tok_per_s": round(tim.get("predicted_per_second", 0), 2),
            "answer_tail": content[-500:],
        })
    return rec


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task"], r["seed"], r["arm"]))
            except Exception:
                pass
        print(f"resuming; {len(done)} results present", flush=True)

    jobs = [(t, s, a) for t in TASKS for s in SEEDS for a in ARMS
            if (t[0], s, a[0]) not in done]
    print(f"{len(jobs)} jobs to run", flush=True)

    t0 = time.time()
    n = 0
    lock = __import__("threading").Lock()
    with OUT.open("a") as out:
        def work(job):
            nonlocal n
            r = call(*job)
            with lock:
                n += 1
                out.write(json.dumps(r) + "\n")
                out.flush()
                if r.get("error"):
                    print(f"[{n}/{len(jobs)}] {r['task']:14s} {r['arm']:14s} "
                          f"ERROR {r['error'][:60]}", flush=True)
                else:
                    print(f"[{n}/{len(jobs)}] {r['task']:14s} {r['arm']:14s} "
                          f"{'OK ' if r['correct'] else 'XX '} "
                          f"got={str(r['got']):>8s} want={r['expected']:<8d} "
                          f"tok={r['completion_tokens']:<6d} "
                          f"think_ch={r['reasoning_chars']:<7d} "
                          f"guillo={str(r['guillotine']):5s} "
                          f"maxtok={str(r['hit_max_tokens']):5s} "
                          f"{r['elapsed_s']:6.0f}s "
                          f"({(time.time()-t0)/60:.0f}m)", flush=True)
            return r

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(work, jobs))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check (a) per-request thinking_budget_tokens enforcement, (b) real concurrency."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-27b-q6-mtp-250k"

HARD = ("How many strings of length 10 over the alphabet {1,2,3} have no two "
        "adjacent characters equal and do not contain '123' as a contiguous "
        "substring? Reason carefully and verify your count.")


def call(effort, budget, prompt=HARD, max_tokens=16384, seed=11):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": seed,
        "thinking_budget_tokens": budget,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    el = time.time() - t0
    m = d["choices"][0]["message"]
    rc = m.get("reasoning_content") or ""
    return {
        "effort": effort, "budget": budget, "elapsed": el,
        "completion_tokens": d["usage"]["completion_tokens"],
        "think_chars": len(rc),
        "guillotine": "Time to stop thinking" in rc,
        "finish": d["choices"][0]["finish_reason"],
        "tok_s": d.get("timings", {}).get("predicted_per_second", 0),
    }


print("=== 1. budget enforcement (tiny budget must trigger guillotine) ===")
for bud in (300, 8192):
    r = call("xhigh", bud)
    print(f"  budget={bud:<8} think_chars={r['think_chars']:<7} "
          f"guillotine={r['guillotine']!s:<6} completion_tok={r['completion_tokens']:<6} "
          f"{r['elapsed']:.1f}s  {r['tok_s']:.1f} tok/s")

print("\n=== 2. concurrency: 4 identical requests at once ===")
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(call, "xhigh", 300, HARD, 16384, 100 + i) for i in range(4)]
    res = [f.result() for f in futs]
wall = time.time() - t0
serial_sum = sum(r["elapsed"] for r in res)
print(f"  wall={wall:.1f}s   sum-of-individual={serial_sum:.1f}s   "
      f"speedup={serial_sum/wall:.2f}x")
for r in res:
    print(f"    tok={r['completion_tokens']:<6} {r['elapsed']:.1f}s "
          f"{r['tok_s']:.1f} tok/s")

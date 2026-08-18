#!/usr/bin/env python3
"""Final analysis: A = xhigh + 8192 cutoff  vs  B = medium + unlimited."""
import json
import pathlib
import statistics as st

rows = [json.loads(l) for l in
        (pathlib.Path(__file__).resolve().parent / "results-phase2.jsonl")
        .read_text().splitlines() if l.strip()]
rows = [r for r in rows if not r.get("error")]

A, B = "A_cutoff", "B_medium_free"
pairs = {}
for r in rows:
    pairs.setdefault((r["task"], r["seed"]), {})[r["arm"]] = r
pairs = {k: v for k, v in pairs.items() if len(v) == 2}

ntask = len({k[0] for k in pairs})
nseed = len({k[1] for k in pairs})
print("%d complete A/B pairs (%d tasks x %d seeds)\n" % (len(pairs), ntask, nseed))

hdr = "%-14s %4s  %5s %7s %7s   %5s %7s %7s"
print(hdr % ("task", "seed", "A ok", "A tok", "guillo", "B ok", "B tok", "maxtok"))
for (t, s), d in sorted(pairs.items()):
    a, b = d[A], d[B]
    print(hdr % (t, s, a["correct"], a["completion_tokens"], a["guillotine"],
                 b["correct"], b["completion_tokens"], b["hit_max_tokens"]))

n = len(pairs)
ta = sum(d[A]["completion_tokens"] for d in pairs.values())
tb = sum(d[B]["completion_tokens"] for d in pairs.values())
ca = sum(d[A]["correct"] for d in pairs.values())
cb = sum(d[B]["correct"] for d in pairs.values())
noans_a = sum(1 for d in pairs.values() if d[A]["got"] is None)
noans_b = sum(1 for d in pairs.values() if d[B]["got"] is None)
cap_a = sum(d[A]["hit_max_tokens"] for d in pairs.values())
cap_b = sum(d[B]["hit_max_tokens"] for d in pairs.values())

row = "%-24s %16s %16s"
print("\n" + row % ("", "A: xhigh+8192", "B: medium free"))
print(row % ("accuracy", "%d/%d" % (ca, n), "%d/%d" % (cb, n)))
print(row % ("total completion tok", ta, tb))
print(row % ("mean tok/task", "%.0f" % (ta / n), "%.0f" % (tb / n)))
print(row % ("median tok/task",
             "%.0f" % st.median([d[A]["completion_tokens"] for d in pairs.values()]),
             "%.0f" % st.median([d[B]["completion_tokens"] for d in pairs.values()])))
print(row % ("no answer produced", noans_a, noans_b))
print(row % ("hit 16384 ceiling", cap_a, cap_b))
print("\nB costs %+.0f%% tokens vs A, for %s accuracy"
      % (100 * (tb - ta) / ta, "identical" if ca == cb else "different"))

aw = sum(1 for d in pairs.values() if d[A]["correct"] and not d[B]["correct"])
bw = sum(1 for d in pairs.values() if d[B]["correct"] and not d[A]["correct"])
both = sum(1 for d in pairs.values() if d[A]["correct"] and d[B]["correct"])
neither = n - aw - bw - both
print("\nPaired outcomes: A-only=%d  B-only=%d  both=%d  neither=%d"
      % (aw, bw, both, neither))

inf = {k: d for k, d in pairs.items() if d[A]["guillotine"]}
print("\n--- Subset where the 8192 cutoff ACTUALLY FIRED (n=%d of %d) ---"
      % (len(inf), n))
print("Only this subset can test the claim; elsewhere the cutoff is inert.")
for (t, s), d in sorted(inf.items()):
    a, b = d[A], d[B]
    va = "correct" if a["correct"] else "WRONG(%s)" % a["got"]
    vb = "correct" if b["correct"] else "WRONG(%s)" % b["got"]
    print("  %-14s seed=%d   A: %-14s %6d tok  |  B: %-14s %6d tok"
          % (t, s, va, a["completion_tokens"], vb, b["completion_tokens"]))
if inf:
    ia = sum(d[A]["correct"] for d in inf.values())
    ib = sum(d[B]["correct"] for d in inf.values())
    hurt = sum(1 for d in inf.values() if d[B]["correct"] and not d[A]["correct"])
    print("\n  accuracy when cutoff fired:  A=%d/%d   B=%d/%d"
          % (ia, len(inf), ib, len(inf)))
    print("  cases where the cutoff LOST an answer medium got right: %d" % hurt)

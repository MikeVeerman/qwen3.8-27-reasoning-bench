#!/usr/bin/env python3
"""Round 3 analysis: three arms, so the cutoff can finally be isolated."""
import json
import pathlib

A, B, D = "A_cutoff", "B_medium_free", "D_medium_cut"
rows = [json.loads(l) for l in
        (pathlib.Path(__file__).resolve().parent / "results-round3.jsonl")
        .read_text().splitlines() if l.strip()]
rows = [r for r in rows if not r.get("error")]

cells = {}
for r in rows:
    cells.setdefault((r["task"], r["seed"]), {})[r["arm"]] = r
full = {k: v for k, v in cells.items() if len(v) == 3}

print("%d complete 3-arm cells (%d tasks x %d seeds)\n"
      % (len(full), len({k[0] for k in full}), len({k[1] for k in full})))

hdr = "%-16s %4s | %-22s | %-22s | %-22s"
print(hdr % ("task", "seed", "A xhigh+8192", "B medium+unlim", "D medium+8192"))
print("-" * 96)


def cell(r):
    v = "OK " if r["correct"] else "XX "
    g = "G" if r["guillotine"] else " "
    m = "!" if r["hit_max_tokens"] else " "
    return "%s%6s tok %-7s %s%s" % (v, r["completion_tokens"], "", g, m)


for (t, s), d in sorted(full.items()):
    print(hdr % (t, s, cell(d[A]), cell(d[B]), cell(d[D])))
print("\n  G = reasoning-budget guillotine fired    ! = hit 16384 max_tokens\n")

n = len(full)
print("%-28s %10s %10s %10s" % ("", "A", "B", "D"))
for label, fn in [
    ("accuracy", lambda r: r["correct"]),
    ("guillotine fired", lambda r: r["guillotine"]),
    ("hit 16384 ceiling", lambda r: r["hit_max_tokens"]),
    ("no answer produced", lambda r: r["got"] is None),
]:
    print("%-28s %10s %10s %10s"
          % (label,
             "%d/%d" % (sum(fn(d[A]) for d in full.values()), n),
             "%d/%d" % (sum(fn(d[B]) for d in full.values()), n),
             "%d/%d" % (sum(fn(d[D]) for d in full.values()), n)))
tot = {k: sum(d[k]["completion_tokens"] for d in full.values()) for k in (A, B, D)}
print("%-28s %10d %10d %10d" % ("total completion tokens", tot[A], tot[B], tot[D]))
print("%-28s %10.0f %10.0f %10.0f"
      % ("mean tokens/task", tot[A] / n, tot[B] / n, tot[D] / n))


def paired(x, y, xn, yn):
    xw = sum(1 for d in full.values() if d[x]["correct"] and not d[y]["correct"])
    yw = sum(1 for d in full.values() if d[y]["correct"] and not d[x]["correct"])
    both = sum(1 for d in full.values() if d[x]["correct"] and d[y]["correct"])
    print("\n%s vs %s" % (xn, yn))
    print("  %s-only=%d   %s-only=%d   both=%d   neither=%d"
          % (xn, xw, yn, yw, both, n - xw - yw - both))
    return xw, yw


print("\n" + "=" * 60)
print("THE ISOLATION: B vs D  (same effort=medium, ONLY the cutoff differs)")
print("=" * 60)
bw, dw = paired(B, D, "B", "D")
print("  -> cutoff COST an answer: %d" % bw)
print("  -> cutoff SAVED an answer: %d" % dw)
print("  tokens: B=%d  D=%d  (%+.0f%% for removing the cutoff)"
      % (tot[B], tot[D], 100 * (tot[B] - tot[D]) / tot[D]))

paired(A, D, "A", "D")   # isolates the effort prompt at constant budget
paired(A, B, "A", "B")   # the deployment choice

# where the cutoff actually fired
fired = {k: d for k, d in full.items() if d[A]["guillotine"] or d[D]["guillotine"]}
print("\n--- cells where the guillotine fired in A or D (n=%d of %d) ---"
      % (len(fired), n))
for (t, s), d in sorted(fired.items()):
    print("  %-16s seed=%d  A=%-4s%s  B=%-4s%s  D=%-4s%s"
          % (t, s,
             "OK" if d[A]["correct"] else "XX", "(G)" if d[A]["guillotine"] else "   ",
             "OK" if d[B]["correct"] else "XX", "(!)" if d[B]["hit_max_tokens"] else "   ",
             "OK" if d[D]["correct"] else "XX", "(G)" if d[D]["guillotine"] else "   "))

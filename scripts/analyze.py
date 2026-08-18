#!/usr/bin/env python3
"""Summarise reasoning-effort benchmark results."""
import json
import pathlib
import statistics as st
from collections import defaultdict

ROWS = [
    json.loads(l)
    for l in (pathlib.Path(__file__).resolve().parent / "results.jsonl")
    .read_text()
    .splitlines()
    if l.strip()
]
ROWS = [r for r in ROWS if not r.get("error")]
CONDS = ["xhigh", "medium", "low"]


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return st.mean(v) if v else 0


def pct(rows, key):
    return 100.0 * sum(1 for r in rows if r.get(key)) / len(rows) if rows else 0


print(f"n = {len(ROWS)} graded responses\n")

# ---- overall -------------------------------------------------------------
print("OVERALL")
print(f"{'effort':8s} {'acc%':>6s} {'compl.tok':>10s} {'think.chars':>12s} "
      f"{'ans.chars':>10s} {'sec':>7s} {'budget-hit%':>12s}")
for c in CONDS:
    rs = [r for r in ROWS if r["condition"] == c]
    if not rs:
        continue
    print(f"{c:8s} {pct(rs,'correct'):6.1f} {agg(rs,'completion_tokens'):10.0f} "
          f"{agg(rs,'reasoning_chars'):12.0f} {agg(rs,'answer_chars'):10.0f} "
          f"{agg(rs,'elapsed_s'):7.1f} {pct(rs,'budget_hit'):12.1f}")

# ---- by category ---------------------------------------------------------
cats = []
for r in ROWS:
    if r["category"] not in cats:
        cats.append(r["category"])

print("\nACCURACY BY CATEGORY (%)")
hdr = f"{'category':10s}" + "".join(f"{c:>10s}" for c in CONDS) + f"{'n/cond':>9s}"
print(hdr)
for cat in cats:
    line = f"{cat:10s}"
    n = 0
    for c in CONDS:
        rs = [r for r in ROWS if r["category"] == cat and r["condition"] == c]
        n = max(n, len(rs))
        line += f"{pct(rs,'correct'):10.1f}" if rs else f"{'-':>10s}"
    print(line + f"{n:9d}")

print("\nMEAN COMPLETION TOKENS BY CATEGORY")
print(hdr)
for cat in cats:
    line = f"{cat:10s}"
    n = 0
    for c in CONDS:
        rs = [r for r in ROWS if r["category"] == cat and r["condition"] == c]
        n = max(n, len(rs))
        line += f"{agg(rs,'completion_tokens'):10.0f}" if rs else f"{'-':>10s}"
    print(line + f"{n:9d}")

print("\nMEAN THINKING CHARS BY CATEGORY")
print(hdr)
for cat in cats:
    line = f"{cat:10s}"
    n = 0
    for c in CONDS:
        rs = [r for r in ROWS if r["category"] == cat and r["condition"] == c]
        n = max(n, len(rs))
        line += f"{agg(rs,'reasoning_chars'):10.0f}" if rs else f"{'-':>10s}"
    print(line + f"{n:9d}")

print("\nMEAN LATENCY (s) BY CATEGORY")
print(hdr)
for cat in cats:
    line = f"{cat:10s}"
    for c in CONDS:
        rs = [r for r in ROWS if r["category"] == cat and r["condition"] == c]
        line += f"{agg(rs,'elapsed_s'):10.1f}" if rs else f"{'-':>10s}"
    print(line)

# ---- per-task accuracy, to spot where they diverge -----------------------
print("\nPER-TASK ACCURACY (correct / samples)")
tasks = []
for r in ROWS:
    if r["task"] not in tasks:
        tasks.append(r["task"])
print(f"{'task':18s}" + "".join(f"{c:>10s}" for c in CONDS))
for t in tasks:
    line = f"{t:18s}"
    for c in CONDS:
        rs = [r for r in ROWS if r["task"] == t and r["condition"] == c]
        line += f"{sum(1 for r in rs if r['correct'])}/{len(rs):<8d}" if rs else f"{'-':>10s}"
    print(line)

# ---- paired comparison: medium vs xhigh on identical (task, seed) --------
print("\nPAIRED medium vs xhigh (same task+seed)")
by = defaultdict(dict)
for r in ROWS:
    by[(r["task"], r["seed"])][r["condition"]] = r
pairs = [v for v in by.values() if "xhigh" in v and "medium" in v]
if pairs:
    win = sum(1 for p in pairs if p["medium"]["correct"] and not p["xhigh"]["correct"])
    loss = sum(1 for p in pairs if p["xhigh"]["correct"] and not p["medium"]["correct"])
    tie = len(pairs) - win - loss
    tok_x = st.mean([p["xhigh"]["completion_tokens"] for p in pairs])
    tok_m = st.mean([p["medium"]["completion_tokens"] for p in pairs])
    sec_x = st.mean([p["xhigh"]["elapsed_s"] for p in pairs])
    sec_m = st.mean([p["medium"]["elapsed_s"] for p in pairs])
    print(f"  pairs={len(pairs)}  medium-wins={win}  xhigh-wins={loss}  ties={tie}")
    print(f"  tokens: xhigh={tok_x:.0f} medium={tok_m:.0f} "
          f"({100*(tok_m-tok_x)/tok_x:+.1f}%)")
    print(f"  latency: xhigh={sec_x:.1f}s medium={sec_m:.1f}s "
          f"({100*(sec_m-sec_x)/sec_x:+.1f}%)")

# ---- truncation / non-answer diagnostics ---------------------------------
print("\nDIAGNOSTICS")
for c in CONDS:
    rs = [r for r in ROWS if r["condition"] == c]
    if not rs:
        continue
    trunc = sum(1 for r in rs if r.get("finish_reason") == "length")
    empty = sum(1 for r in rs if r.get("answer_chars", 0) == 0)
    bud = sum(1 for r in rs if r.get("budget_hit"))
    mx = max((r.get("completion_tokens") or 0) for r in rs)
    print(f"  {c:7s} hit-max-tokens={trunc:3d}  empty-answer={empty:3d}  "
          f"reasoning-budget-guillotine={bud:3d}  max-completion-tok={mx}")

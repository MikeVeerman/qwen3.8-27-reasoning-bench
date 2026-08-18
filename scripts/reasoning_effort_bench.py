#!/usr/bin/env python3
"""
Benchmark: does --chat-template-kwargs '{"reasoning_effort": "medium"}' give a
better quality/verbosity trade-off than the template default (xhigh) on
Qwen3.8-27B?

Conditions are set PER REQUEST via chat_template_kwargs, so all three run
against a single loaded model instance (identical weights, sampler, and
--reasoning-budget 8192 as configured in production).

Results stream to results.jsonl so partial runs are still analysable.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

API = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-27b-q6-mtp-250k"
OUT = pathlib.Path(__file__).resolve().parent / "results.jsonl"

CONDITIONS = ["xhigh", "medium", "low"]  # xhigh == template default
SEEDS = [11, 22, 33]
MAX_TOKENS = 16384

ANSWER_SUFFIX = (
    "\n\nEnd your reply with a final line in exactly this form:\nANSWER: <value>"
)


# --------------------------------------------------------------------------
# answer extraction / normalisation
# --------------------------------------------------------------------------
def answer_line(content):
    """Pull the value off the last 'ANSWER:' line."""
    hits = re.findall(r"ANSWER\s*:\s*(.+)", content or "", re.IGNORECASE)
    return hits[-1].strip() if hits else None


def norm(s):
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("**", "").replace("`", "")
    s = s.replace("\\", "").replace("\u2013", "-").replace("\u2014", "-")
    s = s.strip().rstrip(".").strip()
    return s.lower()


def as_number(s):
    s = norm(s)
    s = re.sub(r"[,\s]", "", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def check_number(expected, tol=1e-9):
    def f(content, _raw):
        v = as_number(answer_line(content))
        return v is not None and abs(v - expected) <= tol

    return f


def check_text(*accepted):
    acc = {a.lower() for a in accepted}

    def f(content, _raw):
        a = norm(answer_line(content))
        return a in acc

    return f


def check_fraction(num, den):
    """Accept 15/128 or its decimal value."""

    def f(content, _raw):
        a = norm(answer_line(content))
        a = re.sub(r"[,\s]", "", a)
        m = re.fullmatch(r"(-?\d+)/(\d+)", a)
        if m:
            return int(m.group(1)) * den == num * int(m.group(2))
        v = as_number(a)
        return v is not None and abs(v - num / den) < 1e-6

    return f


def check_exact_body(expected):
    """Whole reply must be exactly this (format-compliance tasks)."""

    def f(content, _raw):
        return (content or "").strip() == expected

    return f


def check_json_body(expected):
    def f(content, _raw):
        body = (content or "").strip()
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body).strip()
        try:
            return json.loads(body) == expected
        except Exception:
            return False

    return f


# --------------------------------------------------------------------------
# code execution checker
# --------------------------------------------------------------------------
def check_code(func_name, cases):
    """Extract the python block, run it, assert func(*args) == want."""

    def f(content, _raw):
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content or "", re.S)
        if not blocks:
            return False
        code = max(blocks, key=len)
        harness = "\n\n_cases = %r\nfor _a, _w in _cases:\n    _g = %s(*_a)\n    assert _g == _w, (_a, _g, _w)\nprint('PASS')\n" % (
            cases,
            func_name,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code + harness)
            path = fh.name
        try:
            r = subprocess.run(
                [sys.executable, path], capture_output=True, timeout=15, text=True
            )
            return r.returncode == 0 and "PASS" in r.stdout
        except Exception:
            return False
        finally:
            os.unlink(path)

    return f


# --------------------------------------------------------------------------
# task set
# --------------------------------------------------------------------------
def T(tid, cat, prompt, check, suffix=True):
    return {
        "id": tid,
        "category": cat,
        "prompt": prompt + (ANSWER_SUFFIX if suffix else ""),
        "check": check,
    }


TASKS = [
    # --- trivial: thinking here is pure waste -----------------------------
    T("easy_mult", "easy", "What is 17 * 23?", check_number(391)),
    T("easy_capital", "easy", "What is the capital of Australia?",
      check_text("canberra")),
    T("easy_leap", "easy", "How many days are in a leap year?",
      check_number(366)),
    T("easy_percent", "easy", "What is 15% of 200?", check_number(30)),
    T("easy_compare", "easy", "Which number is larger, 9.11 or 9.9?",
      check_text("9.9", "9.90")),
    T("easy_reverse", "easy", "Spell the word 'strawberry' backwards.",
      check_text("yrrebwarts")),

    # --- hard maths: thinking genuinely helps -----------------------------
    T("math_modexp", "math",
      "What is the remainder when 7^100 is divided by 13?", check_number(9)),
    T("math_divis", "math",
      "How many positive integers n <= 1000 are divisible by 3 or by 5 but "
      "not by 15?", check_number(401)),
    T("math_coin", "math",
      "A fair coin is flipped 10 times. What is the probability of getting "
      "exactly 3 heads? Give the answer as a fraction in lowest terms.",
      check_fraction(15, 128)),
    T("math_divisorsum", "math",
      "What is the sum of all positive divisors of 360?", check_number(1170)),
    T("math_table", "math",
      "In how many ways can 8 people be seated around a round table if two "
      "specific people must NOT sit next to each other? Rotations of the same "
      "arrangement are considered identical; reflections are considered "
      "different.", check_number(3600)),
    T("math_nozero", "math",
      "Find the number of ordered pairs of positive integers (a, b) with "
      "a + b = 1000 such that neither a nor b has a zero digit.",
      check_number(738)),

    # --- trick questions: overthinking causes second-guessing -------------
    T("trick_batball", "trick",
      "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
      "the ball. How much does the ball cost, in dollars?",
      check_number(0.05, tol=1e-6)),
    T("trick_rcount", "trick",
      "How many times does the letter 'r' appear in the word 'strawberry'?",
      check_number(3)),
    T("trick_widgets", "trick",
      "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes "
      "would it take 100 machines to make 100 widgets?", check_number(5)),
    T("trick_apples", "trick",
      "I have 3 apples. I eat 2 bananas. How many apples do I have left?",
      check_number(3)),
    T("trick_feathers", "trick",
      "Which weighs more: a pound of feathers or a pound of bricks? Answer "
      "with one of: FEATHERS, BRICKS, SAME.", check_text("same")),

    # --- coding: objectively testable -------------------------------------
    T("code_balanced", "code",
      "Write a Python function `is_balanced(s)` that returns True if the "
      "brackets '()[]{}' in the string s are correctly balanced and nested, "
      "False otherwise. Non-bracket characters are ignored. Return the "
      "function in a single ```python code block.",
      check_code("is_balanced", [
          (("(a[b]{c})",), True), (("([)]",), False), (("",), True),
          (("(((",), False), (("{[()]}",), True), ((")(",), False),
      ]), suffix=False),
    T("code_roman", "code",
      "Write a Python function `roman_to_int(s)` that converts a valid Roman "
      "numeral string (I, V, X, L, C, D, M) to an integer. Return the "
      "function in a single ```python code block.",
      check_code("roman_to_int", [
          (("III",), 3), (("IV",), 4), (("IX",), 9), (("LVIII",), 58),
          (("MCMXCIV",), 1994), (("MMMCMXCIX",), 3999),
      ]), suffix=False),
    T("code_lcp", "code",
      "Write a Python function `longest_common_prefix(strs)` that returns the "
      "longest common prefix string of a list of strings, or '' if there is "
      "none. Return the function in a single ```python code block.",
      check_code("longest_common_prefix", [
          ((["flower", "flow", "flight"],), "fl"),
          ((["dog", "racecar", "car"],), ""),
          (([],), ""), ((["abc"],), "abc"),
          ((["", "abc"],), ""),
      ]), suffix=False),
    T("code_merge", "code",
      "Write a Python function `merge_intervals(intervals)` that takes a list "
      "of [start, end] pairs and returns the list of merged, non-overlapping "
      "intervals sorted by start. Touching intervals such as [1,2] and [2,3] "
      "merge into [1,3]. Return the function in a single ```python code block.",
      check_code("merge_intervals", [
          (([[1, 3], [2, 6], [8, 10], [15, 18]],), [[1, 6], [8, 10], [15, 18]]),
          (([[1, 4], [4, 5]],), [[1, 5]]),
          (([],), []),
          (([[5, 6], [1, 2]],), [[1, 2], [5, 6]]),
      ]), suffix=False),

    # --- format compliance: overthinking causes drift ---------------------
    T("fmt_word", "format",
      "Reply with exactly the word BANANA in uppercase and nothing else. No "
      "punctuation, no explanation.", check_exact_body("BANANA"), suffix=False),
    T("fmt_primes", "format",
      "Output the first 5 prime numbers as a comma-separated list with no "
      "spaces, and nothing else.", check_exact_body("2,3,5,7,11"), suffix=False),
    T("fmt_oneword", "format",
      "Answer with exactly one lowercase word and nothing else: what colour "
      "is a ripe banana?", check_exact_body("yellow"), suffix=False),
    T("fmt_json", "format",
      "Return only a JSON object with the keys \"sum\" and \"product\" giving "
      "the sum and product of the numbers 7 and 6. No markdown fences, no "
      "commentary.", check_json_body({"sum": 13, "product": 42}), suffix=False),
]


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def request(payload, timeout=2400):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task"], r["condition"], r["seed"]))
            except Exception:
                pass
        print(f"resuming, {len(done)} results already present", flush=True)

    total = len(TASKS) * len(SEEDS) * len(CONDITIONS)
    n = 0
    t0 = time.time()
    with OUT.open("a") as out:
        for task in TASKS:
            for seed in SEEDS:
                for cond in CONDITIONS:
                    n += 1
                    key = (task["id"], cond, seed)
                    if key in done:
                        continue
                    payload = {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": task["prompt"]}],
                        "max_tokens": MAX_TOKENS,
                        "seed": seed,
                        "chat_template_kwargs": {"reasoning_effort": cond},
                    }
                    started = time.time()
                    try:
                        resp = request(payload)
                        err = None
                    except Exception as exc:
                        resp, err = None, f"{type(exc).__name__}: {exc}"
                    elapsed = time.time() - started

                    rec = {
                        "task": task["id"], "category": task["category"],
                        "condition": cond, "seed": seed,
                        "elapsed_s": round(elapsed, 2), "error": err,
                    }
                    if resp:
                        msg = resp["choices"][0]["message"]
                        content = msg.get("content") or ""
                        reasoning = msg.get("reasoning_content") or ""
                        usage = resp.get("usage", {})
                        tim = resp.get("timings", {})
                        try:
                            correct = bool(task["check"](content, resp))
                        except Exception:
                            correct = False
                        rec.update({
                            "correct": correct,
                            "finish_reason": resp["choices"][0]["finish_reason"],
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "reasoning_chars": len(reasoning),
                            "answer_chars": len(content),
                            "budget_hit": "Time to stop thinking" in reasoning,
                            "tok_per_s": round(tim.get("predicted_per_second", 0), 2),
                            "answer": content[-600:],
                            "reasoning_head": reasoning[:400],
                        })
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    mark = "?" if err else ("OK " if rec.get("correct") else "XX ")
                    print(
                        f"[{n}/{total}] {mark} {task['id']:16s} {cond:6s} "
                        f"seed={seed} {elapsed:6.1f}s "
                        f"tok={rec.get('completion_tokens')} "
                        f"({(time.time()-t0)/60:.0f}m elapsed)",
                        flush=True,
                    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

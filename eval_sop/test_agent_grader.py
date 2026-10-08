"""Regression tests for the agent-eval grader (no LLM calls).

Usage: python -m eval_sop.test_agent_grader   (exit 0 = all pass)
"""
import sys

from eval_sop.agent_eval import grade_any

DIFF = dict(kind="abs_number", answer=-351,
            q="How many more units of inventory does PART_102 have than PART_167? (A negative number means fewer.)")
STOCK = dict(kind="abs_number", answer=-1464,
             q="By how many units does the current stock of PART_015 exceed (or fall short of) its median 30-day demand forecast?")
STOCK_POS = dict(STOCK, answer=740)

CASES = [
    # (question, model answer, expected grade, description)
    (DIFF, "PART_102 has 351 more units of inventory than PART_167.", False, "wrong direction (qwen H2)"),
    (DIFF, "PART_102 has 351 fewer units of inventory than PART_167.", True, "direction words, correct"),
    (DIFF, "PART_167 has 351 more units of inventory than PART_102.", True,
     "reversed subject, correct (llama H2)"),
    (DIFF, "PART_167 has 351 fewer units than PART_102.", False, "reversed subject, wrong"),
    (DIFF, "The difference is -351 units.", True, "signed number, correct"),
    (DIFF, "The difference is 351 units.", False, "no direction given"),
    (DIFF, "PART_102 has 350 fewer units.", False, "wrong magnitude"),
    (STOCK, "The current stock falls short of the median forecast by 1464 units.", True, "short by, correct"),
    (STOCK, "The current stock exceeds the median forecast by 1464 units.", False, "exceeds, wrong"),
    (STOCK_POS, "The inventory exceeds the median demand by 740 units (1769 - 1029 = 740).", True, "exceeds, correct"),
    (STOCK_POS, "The current stock falls short of the forecast by 740 units.", False, "short, wrong"),
]


def main():
    fails = 0
    for q, ans, exp, desc in CASES:
        got = bool(grade_any(q, ans))
        ok = got == exp
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}: expected {exp}, got {got}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

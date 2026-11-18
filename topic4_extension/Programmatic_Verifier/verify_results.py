"""Verify the archived text edits and independently rescore them in decimal."""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from Self_Consistency.verify_results import answer, correct, rows


def verify(phase):
    base = ROOT.parent / "Direct_CoT"
    out = ROOT / "results" / phase
    q_file = "curated.jsonl" if phase == "pilot" else "expanded_100.jsonl"
    baseline = "pilot_v4" if phase == "pilot" else "formal"
    questions = {q["id"]: q for q in rows(base / "data" / q_file)}
    original = {r["id"]: r["output"] for r in rows(base / "results" / baseline / "records.jsonl") if r["arm"] == "cot"}
    with (out / "paired_scores.csv").open(encoding="utf-8-sig", newline="") as file:
        comparisons = list(csv.DictReader(file))
    all_repairs = rows(out / "repairs.jsonl")
    repairs = {r["id"]: r for r in all_repairs}
    expected = 4 if phase == "pilot" else 100
    if len(original) != expected or len(comparisons) != expected or {r["id"] for r in comparisons} != set(original):
        raise ValueError("Replay coverage is incomplete")
    if len(all_repairs) != len(repairs) or not set(repairs) <= set(original):
        raise ValueError("Unexpected or duplicate repaired outputs")

    totals = Counter()
    changes = Counter()
    for row in comparisons:
        qid = row["id"]
        q, prior = questions[qid], original[qid]
        updated = prior
        if qid in repairs:
            rec = repairs[qid]
            start, end = rec["span_start"], rec["span_end"]
            literal = rec["original_literal"]
            fixed = rec["restored_literal"]
            if prior[start:end] != literal or rec["claim_line"] not in prior[:start]:
                raise ValueError(f"Repair source cannot be traced for {qid}")
            if Decimal(literal).scaleb(rec["exponent"]) != Decimal(fixed):
                raise ValueError(f"Repair changes more than the exponent at {qid}")
            updated = prior[:start] + fixed + prior[end:]
            if updated != rec["repaired_output"] or answer(prior)[1] != answer(updated)[1]:
                raise ValueError(f"A non-numeric field changed at {qid}")

        flags = {}
        for method, text in (("original", prior), ("program", updated)):
            parsed = answer(text)
            for suffix, rtol in (("", q["rtol"]), ("_5pct", Decimal("0.05"))):
                value = correct(parsed, q, rtol)
                if row[f"{method}_correct{suffix}"] != str(value):
                    raise ValueError(f"Independent score mismatch for {qid}/{method}{suffix}")
                totals[method, suffix] += value
                flags[method, suffix] = value
        if row["changed"] != str(qid in repairs):
            raise ValueError(f"Incorrect report of changes at {qid}")
        if qid in repairs:
            changes[int(flags["program", ""]) - int(flags["original", ""])] += 1

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if (summary["repaired_answers"], summary["improved"], summary["worsened"], summary["repaired_unchanged"]) != (len(repairs), changes[1], changes[-1], changes[0]):
        raise ValueError("Incorrect repair counts")
    for (method, suffix), num in totals.items():
        if num != summary["methods"][method][f"correct{suffix}"]:
            raise ValueError("Independent summary mismatch")
    print(f"Verified {expected} answers at both tolerances and all {len(repairs)} text edits.")
    print(f"Correct at 1%: {totals['original', '']} -> {totals['program', '']}; fixed={changes[1]}, broken={changes[-1]}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "formal"], nargs="?", default="formal")
    verify(parser.parse_args().phase)

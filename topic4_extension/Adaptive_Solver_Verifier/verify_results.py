"""Recompute the three policies from raw answers, using decimal scoring."""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import sys

from routing import reasoning_claim

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "Direct_CoT"
sys.path.insert(0, str(ROOT.parent))

from Self_Consistency.verify_results import answer, correct, rows


def unique_by_id(records):
    indexed = {r["id"]: r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate question records")
    return indexed


def verify(phase):
    out = ROOT / "results" / phase
    expected = 4 if phase == "pilot" else 100
    source = BASE / "results" / ("pilot_v4" if phase == "pilot" else "formal") / "records.jsonl"
    q_file = BASE / "data" / ("curated.jsonl" if phase == "pilot" else "expanded_100.jsonl")
    questions = unique_by_id(rows(q_file))
    originals = unique_by_id([r for r in rows(source) if r["arm"] == "cot"])
    reviews = unique_by_id(rows(out / "reviews.jsonl"))
    routes = unique_by_id(rows(out / "routing.jsonl"))
    with (out / "paired_scores.csv").open(encoding="utf-8-sig", newline="") as file:
        pairs = unique_by_id(list(csv.DictReader(file)))
    if len(originals) != expected or set(originals) != set(reviews) or set(reviews) != set(routes) or set(routes) != set(pairs):
        raise ValueError("Incomplete or inconsistent experiment coverage")

    correctness = {policy: [0, 0] for policy in ("original", "always", "adaptive")}
    trigger_counts, review_calls, adopted = Counter(), Counter(), Counter()
    for qid, original in originals.items():
        q, review, recorded = questions[qid], reviews[qid], routes[qid]
        unit = q["target_unit"]
        solver = answer(original["output"])
        revision = answer(review["output"])
        # Independently apply the routing threshold and format gate to read-back values.
        observed = None
        if solver is None:
            trigger = "unparseable"
        elif solver[1] != unit:
            trigger = "unit_mismatch"
        else:
            value, line = reasoning_claim(original["output"], unit)
            observed = Decimal(str(value)) if value is not None else None
            if recorded["claim_line"] != line or recorded["claim_value"] != observed:
                raise ValueError(f"Incorrect draft claim at {qid}")
            trigger = ("reasoning_final_mismatch" if observed is not None
                       and abs(solver[0] - observed) > Decimal("0.01") * max(abs(solver[0]), abs(observed)) else None)
        usable = revision is not None and revision[1] == unit
        if (recorded["trigger"] != trigger or recorded["review_acceptable"] != usable
                or recorded["always_adopted"] != usable or recorded["adaptive_adopted"] != bool(trigger and usable)):
            raise ValueError(f"Routing or adoption mismatch at {qid}")
        trigger_counts[trigger or "no_trigger"] += 1
        thresholds = (q["rtol"], Decimal("0.05"))
        for candidate, row in ((solver, original), (revision, review)):
            expected_scores = [correct(candidate, q, rtol) for rtol in thresholds]
            if expected_scores != [row["score"]["correct"], row["score_5pct"]["correct"]]:
                raise ValueError(f"Independent scoring mismatch at {qid}")

        for policy, call in (("original", False), ("always", True), ("adaptive", bool(trigger))):
            accepted = call and usable
            output = revision if accepted else solver
            evaluated = [correct(output, q, rtol) for rtol in thresholds]
            review_calls[policy] += call
            adopted[policy] += accepted
            row = pairs[qid]
            if ([row[f"{policy}_correct"], row[f"{policy}_correct_5pct"]] != [str(x) for x in evaluated]
                    or row[f"{policy}_adopted"] != str(accepted)):
                raise ValueError(f"Incorrect policy output at {qid}/{policy}")
            for j, b in enumerate(evaluated):
                correctness[policy][j] += b

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if dict(trigger_counts) != summary["triggers"]:
        raise ValueError("Incorrect summary trigger counts")
    for policy, numbers in correctness.items():
        archived = summary["policies"][policy]
        if (numbers != [archived["correct"], archived["correct_5pct"]]
                or adopted[policy] != archived["adopted"] or review_calls[policy] != archived["review_calls"]):
            raise ValueError(f"Independent summary mismatch for {policy}")
    print(f"Verified {expected} original outputs, {expected} reviews, and {expected} routing decisions.")
    print(f"Scores at 1% and 5%: {correctness}")
    print(f"Trigger counts: {dict(trigger_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "formal"], nargs="?", default="formal")
    verify(parser.parse_args().phase)

"""Independently replay the saved scores and votes using decimal arithmetic."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Direct_CoT/data"
RTOL = Decimal("0.01")


def rows(path):
    return [json.loads(line, parse_float=Decimal) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer(text):
    matches = []
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
        try:
            obj = json.loads(match.group(), parse_float=Decimal)
        except json.JSONDecodeError:
            continue
        if "answer" in obj:
            matches.append(obj)
    if len(matches) != 1 or not isinstance(matches[0].get("unit"), str):
        return None
    try:
        number = Decimal(str(matches[0]["answer"]))
    except InvalidOperation:
        return None
    return (number, matches[0]["unit"].strip()) if number.is_finite() else None


def correct(parsed, q, rtol):
    return bool(parsed and parsed[1] == q["target_unit"]
                and abs(parsed[0] - q["gold"]) <= max(q["atol"], abs(q["gold"]) * rtol))


def replay_aggregation(valid):
    if not valid:
        return 1, []
    order = list(valid)
    compatible = {(i, j) for i in order for j in order
                  if abs(valid[i] - valid[j]) <= RTOL * max(abs(valid[i]), abs(valid[j]))}
    groups = [[order[0]]]
    for i in order[1:]:
        candidates = [group for group in groups if all((i, j) in compatible for j in group)]
        if candidates:
            candidates[0].append(i)
        else:
            groups.append([i])
    largest = min(groups, key=lambda group: (-len(group), group[0]))
    return largest[0], groups


def verify(phase):
    out = ROOT / "results" / phase
    qids = json.loads((out / "config.json").read_text(encoding="utf-8"))["question_ids"]
    expected_questions = 100 if phase == "formal" else 4
    if len(qids) != expected_questions or len(set(qids)) != expected_questions:
        raise ValueError("Unexpected question IDs")
    source = DATA / ("expanded_100.jsonl" if phase == "formal" else "curated.jsonl")
    by_id = {q["id"]: q for q in rows(source)}
    samples = defaultdict(dict)
    for row in rows(out / "records.jsonl"):
        qid, k = row["id"], row["sample"]
        if qid not in qids or k in samples[qid] or k not in {1, 2, 3}:
            raise ValueError(f"Unexpected sample at {qid}/{k}")
        parsed = answer(row["output"])
        q = by_id[qid]
        matched = [correct(parsed, q, tolerance) for tolerance in (q["rtol"], Decimal("0.05"))]
        if matched != [row["score"]["correct"], row["score_5pct"]["correct"]]:
            raise ValueError(f"Independent scoring mismatch at {qid}/{k}")
        samples[qid][k] = (parsed, matched)
    archived = rows(out / "vote_records.jsonl")
    votes = {row["id"]: row for row in archived}
    if len(archived) != expected_questions or set(samples) != set(votes) or set(votes) != set(qids):
        raise ValueError("Missing or duplicate questions in samples or votes")
    single, aggregate = [0, 0], [0, 0]
    pairs = Counter()
    coverage = Counter()
    for qid in qids:
        if set(samples[qid]) != {1, 2, 3}:
            raise ValueError(f"Expected three samples for {qid}")
        unit = by_id[qid]["target_unit"]
        parsed = {k: samples[qid][k][0] for k in (1, 2, 3)}
        eligible = {k: result[0] for k, result in parsed.items() if result and result[1] == unit}
        pick, groups = replay_aggregation(eligible)
        vote = votes[qid]
        recorded_groups = [[r["sample"] for r in cluster] for cluster in vote["clusters"]]
        if pick != vote["selected"] or groups != recorded_groups or len(eligible) != vote["eligible"]:
            raise ValueError(f"Independent vote mismatch at {qid}")
        first, selected = samples[qid][1][1], samples[qid][pick][1]
        for i in range(2):
            single[i] += first[i]
            aggregate[i] += selected[i]
        pairs[first[0], selected[0]] += 1
        coverage[sum(samples[qid][k][1][0] for k in (1, 2, 3))] += 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    for numbers, label in ((single, "single_sample"), (aggregate, "majority_vote")):
        if numbers != [summary[label]["correct"], summary[label]["correct_5pct"]]:
            raise ValueError(f"Summary did not reproduce for {label}")
    print(f"Verified {expected_questions * 3} raw samples at both tolerances and {expected_questions} votes.")
    print(f"Single vs vote at 1% and 5%: {single} vs {aggregate}")
    print(f"Paired outcomes (single correct, vote correct): {dict(sorted(pairs.items()))}")
    print(f"Questions by count of correct samples: {dict(sorted(coverage.items()))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "formal"], default="formal", nargs="?")
    verify(parser.parse_args().phase)

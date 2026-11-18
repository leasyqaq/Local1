"""Verify, rescore, and summarize a complete 100-question Direct/CoT run."""

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from run_direct_cot import (
    ARCHIVE_COMMIT, ARMS, DATA_HASH, GENERATION, MODEL_HASH, ROOT, SCORER_HASH, SEED,
    prompt_for, questions_for, read_jsonl, sha256,
)
from scoring import score


def accuracy_interval(correct, n):
    p = correct / n
    z = 1.959963984540054
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [max(0, center - radius), min(1, center + radius)]


def paired_interval(families):
    rng = random.Random(SEED)
    samples = []
    for _ in range(10_000):
        draw = rng.choices(families, k=len(families))
        samples.append(sum(delta for delta, _ in draw) / sum(size for _, size in draw))
    samples.sort()
    return [samples[249], samples[9749]]


def direct_json_only(output):
    try:
        parsed = json.loads(output.strip())
    except ValueError:
        return False
    return isinstance(parsed, dict) and set(parsed) == {"answer", "unit"}


def cot_has_prefix(output):
    """Count text before the answer object as a format check, without judging its reasoning."""
    decoder = json.JSONDecoder()
    for offset, char in enumerate(output):
        if char == "{":
            try:
                parsed, _ = decoder.raw_decode(output[offset:])
            except ValueError:
                continue
            if isinstance(parsed, dict) and "answer" in parsed:
                prefix = re.sub(r"```(?:json)?", "", output[:offset], flags=re.IGNORECASE)
                return any(char.isalpha() for char in prefix)
    return False


def error_type(result):
    if result["correct"]:
        return "correct"
    if not result["format_valid"]:
        return "unparseable"
    if not result["unit_correct"]:
        return "unit_mismatch" if result["numeric_correct"] else "numeric_and_unit_mismatch"
    return "numeric_mismatch"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def review_queue(paired):
    queue = []
    candidates = []
    for row in paired:
        reasons = []
        if row["outcome"] in {"direct_only", "cot_only"}:
            reasons.append("changed_correctness")
        for arm in ARMS:
            for label in ("unparseable", "unit_mismatch"):
                if label in row[f"{arm}_result"]:
                    reasons.append(f"{arm}_{label}")
            if row[f"{arm}_stop"] == "limit":
                reasons.append(f"{arm}_length_limit")
        if reasons:
            queue.append({**row, "review_reason": ";".join(reasons)})
        elif row["outcome"] == "both_wrong":
            candidates.append(row)
    ordered = sorted(candidates, key=lambda row: row["id"])
    sample = random.Random(SEED).sample(ordered, min(10, len(ordered)))
    queue.extend({**row, "review_reason": "sample_both_wrong"} for row in sample)
    return sorted(queue, key=lambda row: row["id"])


def analyze(out):
    questions = {q["id"]: q for q in questions_for("formal")}
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    pinned = {
        "phase": "formal",
        "source_commit": ARCHIVE_COMMIT,
        "expanded_data_sha256": DATA_HASH,
        "model_sha256": MODEL_HASH,
        "scorer_sha256": SCORER_HASH,
        "runner_sha256": sha256(ROOT / "run_direct_cot.py"),
        "question_ids": list(questions),
        "thinking": False,
        "generation": GENERATION,
    }
    if any(config.get(key) != value for key, value in pinned.items()):
        raise ValueError("Run metadata does not match the pinned experiment")
    records = read_jsonl(out / "records.jsonl")
    expected = {(qid, arm) for qid in questions for arm in ARMS}
    by_pair = {(row["id"], row["arm"]): row for row in records}
    if set(by_pair) != expected or len(records) != len(expected):
        raise ValueError("Expected exactly one record per question and method")
    for (qid, arm), row in by_pair.items():
        q = questions[qid]
        if row["prompt"] != prompt_for(q, arm):
            raise ValueError(f"Unexpected prompt for {qid} {arm}")
        if row["score"] != score(row["output"], q) or row["score_5pct"] != score(row["output"], q, rtol=.05):
            raise ValueError(f"Score mismatch for {qid} {arm}")
        if row["output_tokens"] < 0 or row["output_tokens"] > GENERATION["n_predict"] + 4:
            raise ValueError(f"Unexpected output length for {qid} {arm}")

    n = len(questions)
    methods = {}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        correct = sum(row["score"]["correct"] for row in rows)
        methods[arm] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
            "wilson_95": accuracy_interval(correct, n),
            "correct_5pct": sum(row["score_5pct"]["correct"] for row in rows),
            "format_valid": sum(row["score"]["format_valid"] for row in rows),
            "unit_correct": sum(row["score"]["unit_correct"] for row in rows),
            "length_limit": sum(row["stop_type"] == "limit" for row in rows),
            "mean_output_tokens": mean(row["output_tokens"] for row in rows),
            "median_output_tokens": median(row["output_tokens"] for row in rows),
            "mean_seconds": mean(row["seconds"] for row in rows),
            "median_seconds": median(row["seconds"] for row in rows),
            "requested_style_count": sum((direct_json_only if arm == "direct" else cot_has_prefix)(row["output"]) for row in rows),
            "score_outcomes": dict(Counter(error_type(row["score"]) for row in rows)),
        }

    paired = []
    family_deltas = defaultdict(list)
    source_counts = defaultdict(Counter)
    outcomes = Counter()
    for qid, q in questions.items():
        direct = by_pair[qid, "direct"]["score"]
        cot = by_pair[qid, "cot"]["score"]
        d_ok, c_ok = direct["correct"], cot["correct"]
        outcome = "both_correct" if d_ok and c_ok else "direct_only" if d_ok else "cot_only" if c_ok else "both_wrong"
        outcomes[outcome] += 1
        delta = int(c_ok) - int(d_ok)
        family_deltas[q["family_id"]].append(delta)
        source_counts[q["source"]].update({"n": 1, "direct_correct": int(d_ok), "cot_correct": int(c_ok)})
        paired.append({
            "id": qid, "family_id": q["family_id"], "source": q["source"], "domain": q["domain"],
            "gold": q["gold"], "target_unit": q["target_unit"], "outcome": outcome,
            "direct_prediction": direct.get("prediction"), "direct_unit": direct.get("predicted_unit"),
            "direct_stop": by_pair[qid, "direct"]["stop_type"], "direct_result": error_type(direct),
            "cot_prediction": cot.get("prediction"), "cot_unit": cot.get("predicted_unit"),
            "cot_stop": by_pair[qid, "cot"]["stop_type"], "cot_result": error_type(cot),
        })

    discordant = outcomes["direct_only"] + outcomes["cot_only"]
    tail = min(outcomes["direct_only"], outcomes["cot_only"])
    p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(tail + 1)) / 2**discordant)
    summary = {
        "methods": methods,
        "pair_counts": {k: outcomes[k] for k in ("both_correct", "both_wrong", "direct_only", "cot_only")},
        "cot_minus_direct": (outcomes["cot_only"] - outcomes["direct_only"]) / n,
        "paired_family_bootstrap_95": paired_interval([(sum(group), len(group)) for group in family_deltas.values()]),
        "question_families": len(family_deltas),
        "mcnemar_exact_p_question_level": p,
        "by_source": {k: dict(v) for k, v in sorted(source_counts.items())},
    }
    write_csv(out / "paired_scores.csv", paired)
    write_csv(out / "by_source.csv", [{"source": source, **counts} for source, counts in summary["by_source"].items()])
    review = review_queue(paired)
    if review:
        write_csv(out / "error_review_queue.csv", review)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/formal")
    analyze(parser.parse_args().out)

"""Replay the frozen exponent repair on archived SciBench completions."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ADAPTIVE = ROOT.parent / "Adaptive_Solver_Verifier"
sys.path.insert(0, str(ADAPTIVE))

from repair import repair
from run_solver_verifier import paired_interval, questions_for, read_solver, score, sha256


def read_model_reviews(phase, solver_sha, questions):
    out = ADAPTIVE / "results" / phase
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if cfg["solver_records_sha256"] != solver_sha:
        raise ValueError("Model review did not use the same CoT records")
    source = out / "paired_scores.csv"
    with source.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != {q["id"] for q in questions}:
        raise ValueError("Model review coverage differs from replay")
    return by_id, sha256(source)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(phase):
    qlist = questions_for(phase)
    original_config, originals, solver_sha = read_solver(phase, qlist)
    reviews, reviews_sha = read_model_reviews(phase, solver_sha, qlist)
    frozen = {"runner_sha256": sha256(Path(__file__)), "repair_sha256": sha256(ROOT / "repair.py"),
              "routing_sha256": sha256(ADAPTIVE / "routing.py"), "protocol_sha256": sha256(ROOT / "README.md")}
    config = {**frozen, "phase": phase, "source_commit": original_config["source_commit"],
              "expanded_data_sha256": original_config["expanded_data_sha256"],
              "scorer_sha256": original_config["scorer_sha256"], "original_model_sha256": original_config["model_sha256"],
              "solver_records_sha256": solver_sha, "model_review_scores_sha256": reviews_sha,
              "question_ids": [q["id"] for q in qlist]}
    if phase == "formal":
        pilot = ROOT / "results/pilot"
        pilot_config = json.loads((pilot / "config.json").read_text(encoding="utf-8"))
        if any(pilot_config[k] != v for k, v in frozen.items()) or not (pilot / "summary.json").is_file():
            raise ValueError("The frozen protocol must pass replay on the pilot records first")
    out = ROOT / "results" / phase
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / "config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")) != config:
        raise ValueError("Replaying into an output directory with different inputs is not allowed")
    write_json(config_file, config)

    repairs, paired = [], []
    changes, family_deltas = Counter(), defaultdict(list)
    for q in qlist:
        qid, unit = q["id"], q["target_unit"]
        original = originals[qid]["output"]
        updated, evidence = repair(original, unit)
        before, after = score(original, q), score(updated, q)
        before_5, after_5 = score(original, q, rtol=.05), score(updated, q, rtol=.05)
        review = reviews[qid]
        if [str(before["correct"]), str(before_5["correct"])] != [review["original_correct"], review["original_correct_5pct"]]:
            raise ValueError(f"Original scores differ at {qid}")
        delta = int(after["correct"]) - int(before["correct"])
        if evidence:
            changes[delta] += 1
            repairs.append({"id": qid, **evidence, "repaired_output": updated,
                            "original_score": before, "repaired_score": after})
        family_deltas[q["family_id"]].append(delta)
        paired.append({"id": qid, "family_id": q["family_id"], "target_unit": unit, "gold": q["gold"],
                       "cot_prediction": before.get("prediction"), "repaired_prediction": after.get("prediction"),
                       "changed": bool(evidence), "original_correct": before["correct"],
                       "always_correct": review["always_correct"] == "True",
                       "adaptive_correct": review["adaptive_correct"] == "True", "program_correct": after["correct"],
                       "original_correct_5pct": before_5["correct"],
                       "always_correct_5pct": review["always_correct_5pct"] == "True",
                       "adaptive_correct_5pct": review["adaptive_correct_5pct"] == "True",
                       "program_correct_5pct": after_5["correct"],
                       "outcome": "improved" if delta > 0 else "worsened" if delta < 0 else "unchanged"})
    n = len(paired)
    summaries = {}
    for scheme, calls in (("original", 0), ("always", n), ("adaptive", sum(bool(row["trigger"]) for row in reviews.values())), ("program", 0)):
        num = sum(r[f"{scheme}_correct"] for r in paired)
        summaries[scheme] = {"n": n, "correct": num, "accuracy": num / n,
                             "correct_5pct": sum(r[f"{scheme}_correct_5pct"] for r in paired), "extra_model_calls": calls}
    summary = {"questions": n, "methods": summaries, "repaired_answers": len(repairs),
               "improved": changes[1], "worsened": changes[-1], "repaired_unchanged": changes[0],
               "unmodified": n - len(repairs), "change_vs_original": (changes[1] - changes[-1]) / n,
               "paired_family_bootstrap_95": paired_interval([(sum(ds), len(ds)) for ds in family_deltas.values()])}
    with (out / "paired_scores.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (out / "repairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in repairs), encoding="utf-8")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "formal"])
    run(parser.parse_args().phase)

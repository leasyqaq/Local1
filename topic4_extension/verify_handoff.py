"""Verify archived artifacts and presentation numbers without running inference."""

from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

from Self_Consistency.verify_results import answer, correct, rows


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def metrics_rows():
    main_path = "Direct_CoT/results/formal/summary.json"
    sampling_path = "Self_Consistency/results/formal/summary.json"
    review_path = "Adaptive_Solver_Verifier/results/formal/summary.json"
    program_path = "Programmatic_Verifier/results/formal/summary.json"
    main = read_json(main_path)["methods"]
    sampling = read_json(sampling_path)
    review = read_json(review_path)["policies"]
    program = read_json(program_path)["methods"]["program"]
    specs = [
        ("main", "Direct Answer", main["direct"], 1, main_path),
        ("main", "Greedy CoT", main["cot"], 1, main_path),
        ("sampling", "Sampled CoT (first sample)", sampling["single_sample"], 1, sampling_path),
        ("sampling", "Self-Consistency (3 samples)", sampling["majority_vote"], 3, sampling_path),
        ("verification", "Always review", review["always"], 2, review_path),
        ("verification", "Adaptive review", review["adaptive"], 1.07, review_path),
        ("verification", "Programmatic exponent repair", program, 1, program_path),
    ]
    for group, name, metrics, calls, source in specs:
        yield {"comparison": group, "method": name, "questions": metrics["n"],
               "correct_primary": metrics["correct"], "accuracy_percent": round(metrics["accuracy"] * 100, 4),
               "correct_5pct": metrics["correct_5pct"], "model_calls_per_question": calls,
               "mean_generated_tokens_per_question": metrics.get("mean_output_tokens", main["cot"]["mean_output_tokens"]),
               "mean_inference_seconds_per_question": round(metrics["mean_seconds"], 4) if "mean_seconds" in metrics else "",
               "evidence_json": source}


def verify_manifest():
    files = read_json("manifest.json")["files"]
    for record in files:
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Changed or missing artifact: {record['path']}")
    print(f"Manifest verified: {len(files)} files", flush=True)


def verify_main():
    out = ROOT / "Direct_CoT/results/formal"
    questions = {q["id"]: q for q in rows(ROOT / "Direct_CoT/data/expanded_100.jsonl") if q["variant"] == "original"}
    raw = rows(out / "records.jsonl")
    pairs = {(r["id"], r["arm"]): r for r in raw}
    expected = {(qid, arm) for qid in questions for arm in ("direct", "cot")}
    if len(questions) != 100 or len(raw) != 200 or set(pairs) != expected:
        raise ValueError("The main run must have exactly one record per question and arm")
    by_method = Counter()
    for (qid, arm), row in pairs.items():
        q = questions[qid]
        parsed = answer(row["output"])
        for key, tolerance in (("score", q["rtol"]), ("score_5pct", Decimal("0.05"))):
            evaluated = correct(parsed, q, tolerance)
            if row[key]["correct"] != evaluated:
                raise ValueError(f"Independent scoring mismatch at {qid}/{arm}/{key}")
            by_method[arm, key] += evaluated
    summary = read_json("Direct_CoT/results/formal/summary.json")["methods"]
    for arm, metrics in summary.items():
        if (metrics["correct"], metrics["correct_5pct"]) != (by_method[arm, "score"], by_method[arm, "score_5pct"]):
            raise ValueError(f"Main scores differ for {arm}")
    print(f"Main run: 200 original outputs re-scored; Direct={by_method['direct','score']}, CoT={by_method['cot','score']}", flush=True)


def main():
    verify_manifest()
    verify_main()
    for bonus in ("Self_Consistency", "Adaptive_Solver_Verifier", "Programmatic_Verifier"):
        subprocess.run([sys.executable, "-B", str(ROOT / bonus / "verify_results.py"), "formal"], check=True)
    with (ROOT / "ppt_metrics.csv").open(encoding="utf-8-sig", newline="") as file:
        comparison = list(csv.DictReader(file))
    if comparison != [{k: str(v) for k, v in row.items()} for row in metrics_rows()]:
        raise ValueError("Presentation data differs from the archived summaries")
    print("Presentation data verified against the formal summaries", flush=True)


if __name__ == "__main__":
    main()

"""Benchmark a frozen CoT baseline, unconditional verifier, and selective verifier."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import socket
from statistics import mean
import subprocess
import sys
import time

from routing import adopt, route

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "Direct_CoT"
sys.path.insert(0, str(BASE))

from analyze_results import accuracy_interval, paired_interval
from run_direct_cot import (
    ANSWER_FORMAT, COMMON, CONTEXT, GENERATION, MODEL_HASH, SYSTEM, launch_args,
    prompt_for, questions_for, read_jsonl, request, sha256, wait_until_ready,
)
from scoring import parse_answer, score


def json_file(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_solver(phase, questions):
    origin = BASE / "results" / ("pilot_v4" if phase == "pilot" else "formal")
    cfg = json.loads((origin / "config.json").read_text(encoding="utf-8"))
    source = origin / "records.jsonl"
    rows = [r for r in read_jsonl(source) if r["arm"] == "cot"]
    by_id = {r["id"]: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != {q["id"] for q in questions}:
        raise ValueError("Unexpected baseline question IDs or duplicate solutions")
    if cfg["runner_sha256"] != sha256(BASE / "run_direct_cot.py") or cfg["generation"] != GENERATION:
        raise ValueError("Original generation settings do not match the archived CoT run")
    for q in questions:
        solver = by_id[q["id"]]
        if solver["prompt"] != prompt_for(q, "cot") or solver["score"] != score(solver["output"], q):
            raise ValueError(f"Cannot verify the original CoT record for {q['id']}")
    return cfg, by_id, sha256(source)


def review_prompt(question, draft):
    unit = json.dumps(question["target_unit"], ensure_ascii=False)
    user = (
        f"{COMMON}\n\nProblem:\n{question['question']}\n\n"
        f"Candidate solution to audit:\n<draft>\n{draft}\n</draft>\n\n"
        f"{ANSWER_FORMAT.format(unit=unit)}\n"
        "Independently verify the candidate against the problem. Begin with 'Step 1:' and use two to four short steps. "
        "Check the formula, given values, arithmetic, units and powers of ten. "
        "If the draft is wrong or incomplete, recalculate the answer. "
        "Finish with your verified numeric answer in the required plain-text JSON object."
    )
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def verify_records(rows, questions, originals, complete=False):
    keys = {q["id"] for q in questions}
    found = {row["id"]: row for row in rows}
    if len(rows) != len(found) or not set(found) <= keys or (complete and set(found) != keys):
        raise ValueError("Invalid question coverage in verifier records")
    for q in questions:
        qid = q["id"]
        if qid in found:
            r = found[qid]
            if (r["prompt"] != review_prompt(q, originals[qid]["output"])
                    or r["score"] != score(r["output"], q)
                    or r["score_5pct"] != score(r["output"], q, rtol=.05)):
                raise ValueError(f"Verifier prompt or score mismatch for {qid}")
    return found


def generate(port, q, draft):
    prompt = review_prompt(q, draft)
    tokens = request(port, "tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    if not tokens or len(tokens) + GENERATION["n_predict"] > CONTEXT:
        raise ValueError(f"Invalid verifier context length for {q['id']}")
    start = time.perf_counter()
    reply = request(port, "completion", {**GENERATION, "prompt": tokens})
    elapsed = time.perf_counter() - start
    output = reply["content"]
    if reply.get("truncated") or not isinstance(output, str) or reply.get("stop_type") not in {"eos", "word", "limit"}:
        raise ValueError(f"Unexpected verifier response for {q['id']}")
    return {"id": q["id"], "prompt": prompt, "output": output, "input_tokens": len(tokens),
            "output_tokens": reply["tokens_predicted"], "stop_type": reply["stop_type"],
            "seconds": round(elapsed, 6), "timings": reply.get("timings"),
            "score": score(output, q), "score_5pct": score(output, q, rtol=.05)}


def run(out, questions, originals, cfg):
    file = out / "reviews.jsonl"
    rows = read_jsonl(file) if file.exists() else []
    completed = verify_records(rows, questions, originals)
    if len(completed) == len(questions):
        return rows
    runtime = BASE / "_runtime"
    model = runtime / "qwen3-1.7b-Q4_K_M.gguf"
    exe = runtime / "llama-b12404-vulkan/llama-server.exe"
    if sha256(model) != MODEL_HASH or sha256(exe) != cfg["backend_sha256"]:
        raise ValueError("Model or server differs from the original CoT run")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (out / "server.log").open("ab") as log, file.open("a", encoding="utf-8") as output:
        proc = subprocess.Popen(launch_args(exe, model, cfg["device"], cfg["gpu_layers"], port),
                                stdout=log, stderr=log, cwd=exe.parent,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            wait_until_ready(port, proc)
            for q in questions:
                qid = q["id"]
                if qid in completed:
                    continue
                row = generate(port, q, originals[qid]["output"])
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                rows.append(row)
                print(f"{len(rows)}/{len(questions)} {qid} review_tokens={row['output_tokens']}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return rows


def metrics(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {"n": n, "correct": correct, "accuracy": correct / n, "wilson_95": accuracy_interval(correct, n),
            "correct_5pct": sum(r["correct_5pct"] for r in rows),
            "review_calls": sum(r["review_call"] for r in rows), "adopted": sum(r["adopted"] for r in rows),
            **{f"mean_{k}": mean(r[k] for r in rows) for k in ("input_tokens", "output_tokens", "seconds")},
            **{f"extra_mean_{k}": mean(r[f"extra_{k}"] for r in rows) for k in ("input_tokens", "output_tokens", "seconds")}}


def analyze(out, questions, originals, reviews):
    by_id = verify_records(reviews, questions, originals, complete=True)
    policies = {policy: [] for policy in ("original", "always", "adaptive")}
    cost_names = ("input_tokens", "output_tokens", "seconds")
    flags, routes, csv_rows = Counter(), [], []
    for q in questions:
        qid, unit = q["id"], q["target_unit"]
        original, review = originals[qid], by_id[qid]
        routing = route(original["output"], unit, parse_answer)
        trigger = routing["trigger"]
        acceptable = adopt(review["output"], unit, parse_answer)
        flags[trigger or "no_trigger"] += 1
        routes.append({"id": qid, **routing, "review_acceptable": acceptable,
                       "always_adopted": acceptable, "adaptive_adopted": bool(trigger) and acceptable})
        paired = {"id": qid, "family_id": q["family_id"], "gold": q["gold"], "target_unit": unit,
                  "trigger": trigger, "claim_value": routing["claim_value"], "review_acceptable": acceptable,
                  "review_stop_type": review["stop_type"]}
        for policy, call in (("original", False), ("always", True), ("adaptive", bool(trigger))):
            adopted = call and acceptable
            chosen = review if adopted else original
            row = {"correct": score(chosen["output"], q)["correct"],
                   "correct_5pct": score(chosen["output"], q, rtol=.05)["correct"],
                   "review_call": call, "adopted": adopted}
            for cost in cost_names:
                row[f"extra_{cost}"] = review[cost] if call else 0
                row[cost] = original[cost] + row[f"extra_{cost}"]
            policies[policy].append(row)
            paired[f"{policy}_correct"] = row["correct"]
            paired[f"{policy}_correct_5pct"] = row["correct_5pct"]
            paired[f"{policy}_prediction"] = chosen["score"].get("prediction")
            paired[f"{policy}_adopted"] = adopted
        csv_rows.append(paired)

    comparisons = {}
    for tested, control in (("always", "original"), ("adaptive", "original"), ("adaptive", "always")):
        families, outcomes = defaultdict(list), Counter()
        for q, a, b in zip(questions, policies[tested], policies[control], strict=True):
            delta = int(a["correct"]) - int(b["correct"])
            families[q["family_id"]].append(delta)
            outcomes[delta] += 1
        comparisons[f"{tested}_vs_{control}"] = {
            "improved_questions": outcomes[1], "worsened_questions": outcomes[-1],
            "delta": (outcomes[1] - outcomes[-1]) / len(questions),
            "paired_family_bootstrap_95": paired_interval([(sum(f), len(f)) for f in families.values()]),
        }
    summary = {"questions": len(questions), "recorded_reviews": len(reviews), "triggers": dict(flags),
               "parsed_claims": sum(r["claim_value"] is not None for r in routes),
               "review_format_valid": sum(r["score"]["format_valid"] for r in reviews),
               "review_unit_correct": sum(r["score"]["unit_correct"] for r in reviews),
               "review_length_limit": sum(r["stop_type"] == "limit" for r in reviews),
               "policies": {name: metrics(rows) for name, rows in policies.items()}, "comparisons": comparisons}
    if out.name == "pilot" and summary["policies"]["always"]["adopted"] < 3:
        raise ValueError("Pilot requires at least three parseable, unit-correct reviews")
    with (out / "paired_scores.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (out / "routing.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in routes), encoding="utf-8")
    json_file(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def configuration(phase, questions, baseline, solver_sha):
    keys = ("source_commit", "expanded_data_sha256", "scorer_sha256", "model", "model_sha256",
            "backend_version", "backend_sha256", "platform", "device", "gpu_layers", "threads",
            "context", "thinking", "generation")
    return {**{key: baseline[key] for key in keys}, "phase": phase, "solver_records_sha256": solver_sha,
            "solver_runner_sha256": baseline["runner_sha256"], "runner_sha256": sha256(Path(__file__)),
            "routing_sha256": sha256(ROOT / "routing.py"), "protocol_sha256": sha256(ROOT / "README.md"),
            "question_ids": [q["id"] for q in questions]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "formal"])
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    questions = questions_for(args.phase)
    baseline, originals, solver_sha = read_solver(args.phase, questions)
    config = configuration(args.phase, questions, baseline, solver_sha)
    out = ROOT / "results" / args.phase
    if args.phase == "formal" and not args.analysis_only:
        pilot_questions = questions_for("pilot")
        pilot_cfg, _, pilot_sha = read_solver("pilot", pilot_questions)
        pilot = ROOT / "results/pilot"
        if (not (pilot / "summary.json").is_file()
                or json.loads((pilot / "config.json").read_text(encoding="utf-8"))
                != configuration("pilot", pilot_questions, pilot_cfg, pilot_sha)):
            raise ValueError("This version of the verifier must pass the pilot first")
    cfg_path = out / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Run settings changed after config was frozen")
    elif not args.analysis_only:
        out.mkdir(parents=True, exist_ok=True)
        json_file(cfg_path, config)
    else:
        raise ValueError("Cannot analyze a missing run")
    reviews = read_jsonl(out / "reviews.jsonl") if args.analysis_only else run(out, questions, originals, config)
    analyze(out, questions, originals, reviews)


if __name__ == "__main__":
    main()

"""Run and verify three-sample CoT voting without modifying the original run."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import socket
from statistics import mean
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "Direct_CoT"
sys.path.insert(0, str(BASE))

from analyze_results import accuracy_interval, paired_interval
from run_direct_cot import (
    CONTEXT, GENERATION, MODEL_HASH, launch_args, prompt_for, questions_for, read_jsonl,
    request, sha256, wait_until_ready,
)
from scoring import parse_answer, score


SAMPLES = (1, 2, 3)
SETTINGS = {key: value for key, value in GENERATION.items() if key != "seed"}
SETTINGS.update(temperature=0.7, top_k=20, top_p=0.8, min_p=0.0)
VOTE_RTOL = 0.01


def sample_seed(qid, sample):
    material = f"SDSC6020-SC-v1|{qid}|{sample}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7fffffff


def select_vote(rows, unit):
    clusters = []
    excluded = {}
    for row in rows:
        parsed, error = parse_answer(row["output"])
        if error or parsed["unit"].strip() != unit:
            excluded[str(row["sample"])] = error or "unit mismatch"
            continue
        value = parsed["answer"]
        for cluster in clusters:
            if all(abs(value - old) <= VOTE_RTOL * max(abs(value), abs(old)) for _, old in cluster):
                cluster.append((row["sample"], value))
                break
        else:
            clusters.append([(row["sample"], value)])
    if not clusters:
        return {"selected": rows[0]["sample"], "support": 0, "outcome": "no_eligible_answer",
                "clusters": [], "excluded": excluded}
    winner = max(clusters, key=len)
    return {"selected": winner[0][0], "support": len(winner),
            "outcome": "majority" if len(winner) >= 2 else "first_eligible",
            "clusters": [[{"sample": index, "value": value} for index, value in group] for group in clusters],
            "excluded": excluded}


def json_file(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_baseline(questions):
    baseline_config = json.loads((BASE / "results/formal/config.json").read_text(encoding="utf-8"))
    if baseline_config["runner_sha256"] != sha256(BASE / "run_direct_cot.py"):
        raise ValueError("Original CoT runner differs from the saved configuration")
    records = read_jsonl(BASE / "results/formal/records.jsonl")
    cot = [r for r in records if r["arm"] == "cot"]
    by_id = {row["id"]: row for row in cot}
    if len(cot) != len(by_id) or set(by_id) != {q["id"] for q in questions}:
        raise ValueError("Original CoT records do not contain the same 100 questions")
    for q in questions:
        r = by_id[q["id"]]
        if r["prompt"] != prompt_for(q, "cot") or r["score"] != score(r["output"], q):
            raise ValueError(f"Greedy CoT comparison failed for {q['id']}")
    return baseline_config, by_id


def config_for(phase, questions, baseline):
    fields = ("source_commit", "expanded_data_sha256", "scorer_sha256", "model", "model_sha256",
              "backend_version", "backend_sha256", "platform", "device", "gpu_layers", "threads",
              "context", "thinking")
    return {**{key: baseline[key] for key in fields}, "phase": phase,
            "runner_sha256": sha256(Path(__file__)), "prompt_runner_sha256": baseline["runner_sha256"],
            "baseline_records_sha256": sha256(BASE / "results/formal/records.jsonl"),
            "protocol_sha256": sha256(ROOT / "README.md"), "generation": SETTINGS,
            "sample_numbers": SAMPLES, "vote_relative_tolerance": VOTE_RTOL,
            "question_ids": [q["id"] for q in questions]}


def check_records(rows, questions, complete=False):
    expected = {(q["id"], sample) for q in questions for sample in SAMPLES}
    by_key = {(row["id"], row["sample"]): row for row in rows}
    if len(rows) != len(by_key) or not set(by_key) <= expected or (complete and set(by_key) != expected):
        raise ValueError("Records include duplicate, missing, or unexpected samples")
    by_id = {q["id"]: q for q in questions}
    for (qid, sample), row in by_key.items():
        q = by_id[qid]
        if (row["seed"] != sample_seed(qid, sample) or row["prompt"] != prompt_for(q, "cot")
                or row["score"] != score(row["output"], q)
                or row["score_5pct"] != score(row["output"], q, rtol=.05)):
            raise ValueError(f"Prompt, seed, or score mismatch for {qid} sample {sample}")
    return by_key


def generate(port, q, sample, prompt, tokens):
    seed = sample_seed(q["id"], sample)
    start = time.perf_counter()
    result = request(port, "completion", {**SETTINGS, "seed": seed, "prompt": tokens})
    elapsed = time.perf_counter() - start
    output = result["content"]
    if result.get("truncated") or not isinstance(output, str) or result.get("stop_type") not in {"eos", "word", "limit"}:
        raise ValueError(f"Unexpected completion response for {q['id']} sample {sample}")
    return {"id": q["id"], "family_id": q["family_id"], "source": q["source"], "sample": sample,
            "seed": seed, "prompt": prompt, "output": output, "input_tokens": len(tokens),
            "output_tokens": result["tokens_predicted"], "stop_type": result["stop_type"],
            "seconds": round(elapsed, 6), "timings": result.get("timings"),
            "score": score(output, q), "score_5pct": score(output, q, rtol=.05)}


def run(out, questions, config):
    record_path = out / "records.jsonl"
    rows = read_jsonl(record_path) if record_path.exists() else []
    done = check_records(rows, questions)
    total = len(questions) * len(SAMPLES)
    if len(done) == total:
        return rows
    runtime = BASE / "_runtime"
    model = runtime / "qwen3-1.7b-Q4_K_M.gguf"
    exe = runtime / "llama-b12404-vulkan/llama-server.exe"
    if sha256(model) != MODEL_HASH or sha256(exe) != config["backend_sha256"]:
        raise ValueError("Runtime binaries or model do not match the original experiment")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = launch_args(exe, model, config["device"], config["gpu_layers"], port)
    with (out / "server.log").open("ab") as log, record_path.open("a", encoding="utf-8") as log_rows:
        proc = subprocess.Popen(args, stdout=log, stderr=log, cwd=exe.parent,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            wait_until_ready(port, proc)
            for q in questions:
                left = [sample for sample in SAMPLES if (q["id"], sample) not in done]
                if not left:
                    continue
                prompt = prompt_for(q, "cot")
                tokens = request(port, "tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
                if not tokens or len(tokens) + SETTINGS["n_predict"] > CONTEXT:
                    raise ValueError(f"Invalid prompt length for {q['id']}")
                for sample in left:
                    row = generate(port, q, sample, prompt, tokens)
                    log_rows.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    log_rows.flush()
                    rows.append(row)
                    print(f"{len(rows)}/{total} {q['id']} sample={sample} tokens={row['output_tokens']}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return rows


def accuracy(rows, tokens, seconds):
    count = sum(row["score"]["correct"] for row in rows)
    return {"correct": count, "n": len(rows), "accuracy": count / len(rows),
            "wilson_95": accuracy_interval(count, len(rows)),
            "correct_5pct": sum(row["score_5pct"]["correct"] for row in rows),
            "mean_output_tokens": mean(tokens), "mean_seconds": mean(seconds)}


def summarize(out, questions, raw, greedy):
    keyed = check_records(raw, questions, complete=True)
    if len({row["seed"] for row in raw}) != len(raw):
        raise ValueError("Sample seeds must be distinct")
    pair_rows, votes, selected_rows, vote_tokens, vote_seconds = [], [], [], [], []
    family_deltas, outcomes, support = defaultdict(list), Counter(), Counter()
    for q in questions:
        qid = q["id"]
        samples = [keyed[qid, sample] for sample in SAMPLES]
        start = time.perf_counter()
        vote = select_vote(samples, q["target_unit"])
        voting_seconds = time.perf_counter() - start
        pick = keyed[qid, vote["selected"]]
        selected_rows.append(pick)
        support[vote["support"]] += 1
        vote_tokens.append(sum(s["output_tokens"] for s in samples))
        vote_seconds.append(sum(s["seconds"] for s in samples) + voting_seconds)
        vote.update(id=qid, eligible=len(SAMPLES) - len(vote["excluded"]), voting_seconds=voting_seconds)
        votes.append(vote)
        first_ok, vote_ok = samples[0]["score"]["correct"], pick["score"]["correct"]
        outcome = ("both_correct" if first_ok else "vote_only") if vote_ok else ("single_only" if first_ok else "both_wrong")
        outcomes[outcome] += 1
        family_deltas[q["family_id"]].append(int(vote_ok) - int(first_ok))
        row = {"id": qid, "source": q["source"], "family_id": q["family_id"], "gold": q["gold"],
               "target_unit": q["target_unit"], "selected_sample": vote["selected"], "support": vote["support"],
               "eligible": vote["eligible"], "rule": vote["outcome"], "outcome": outcome,
               "vote_correct": vote_ok, "vote_prediction": pick["score"].get("prediction"),
               "vote_output_tokens": vote_tokens[-1], "vote_seconds": vote_seconds[-1]}
        for s in samples:
            row[f"sample_{s['sample']}_correct"] = s["score"]["correct"]
            row[f"sample_{s['sample']}_prediction"] = s["score"].get("prediction")
        if qid in greedy:
            row["greedy_correct"] = greedy[qid]["score"]["correct"]
        pair_rows.append(row)

    def per_run(rows):
        return accuracy(rows, [r["output_tokens"] for r in rows], [r["seconds"] for r in rows])

    summary = {"questions": len(questions), "samples": len(raw),
               "single_sample": per_run([keyed[q["id"], 1] for q in questions]),
               "majority_vote": accuracy(selected_rows, vote_tokens, vote_seconds),
               "accuracy_by_sample": {i: per_run([keyed[q["id"], i] for q in questions]) for i in SAMPLES},
               "pair_counts": {k: outcomes[k] for k in ("both_correct", "both_wrong", "single_only", "vote_only")},
               "vote_support_counts": {k: support[k] for k in range(4)},
               "unparseable_count": sum(not r["score"]["format_valid"] for r in raw),
               "length_limit_count": sum(r["stop_type"] == "limit" for r in raw),
               "vote_minus_single": (outcomes["vote_only"] - outcomes["single_only"]) / len(questions),
               "paired_family_bootstrap_95": paired_interval([(sum(x), len(x)) for x in family_deltas.values()])}
    if greedy:
        summary["greedy_cot"] = per_run([greedy[q["id"]] for q in questions])
    if out.name == "pilot" and (min(v["eligible"] for v in votes) < 2
                                or all(len({keyed[q["id"], s]["output"] for s in SAMPLES}) == 1 for q in questions)):
        raise ValueError("Pilot did not pass eligibility and diversity checks")
    with (out / "paired_scores.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    (out / "vote_records.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in votes), encoding="utf-8")
    json_file(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "formal"])
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    formal = questions_for("formal")
    baseline, greedy = load_baseline(formal)
    questions = questions_for(args.phase)
    out = ROOT / "results" / args.phase
    cfg = config_for(args.phase, questions, baseline)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
            raise ValueError("Existing configuration is different")
    elif not args.analysis_only:
        out.mkdir(parents=True, exist_ok=True)
        json_file(cfg_path, cfg)
    else:
        raise ValueError("Cannot analyze a missing run")
    if args.phase == "formal" and not args.analysis_only:
        pilot_dir = ROOT / "results/pilot"
        pilot_cfg = config_for("pilot", questions_for("pilot"), baseline)
        if json.loads((pilot_dir / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(pilot_cfg)) or not (pilot_dir / "summary.json").is_file():
            raise ValueError("The current configuration must pass the pilot before formal inference")
    rows = read_jsonl(out / "records.jsonl") if args.analysis_only else run(out, questions, cfg)
    summarize(out, questions, rows, greedy if args.phase == "formal" else {})


if __name__ == "__main__":
    main()

"""Run a paired Qwen3 Direct Answer vs. CoT comparison with llama.cpp."""

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from scoring import score


ROOT = Path(__file__).resolve().parent
ARCHIVE_COMMIT = "33139f48c4eab8cd94b0531d7895d50bada4da59"
MODEL_HASH = "3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
DATA_HASH = "927a62cddb73f04e87c338255e9cd816340959231787db3aab20d6ea9c5243e7"
SCORER_HASH = "9f22ff722f843cf00714ac03dbdedbdf26fa9aec3b002ffd48ad7708971ec545"
PILOT_IDS = {"atkins_002", "atkins_007", "atkins_012", "atkins_013"}
SEED = 20261118
CONTEXT = 4096
THREADS = 4
SYSTEM = (
    "You solve university scientific calculations. Follow the requested output format. "
    "Do not assume access to reference answers. Use the stated assumptions."
)
COMMON = """Use R=8.31446261815324 J/(mol K), hc/k=1.4387768775039338 cm K when needed.
For liquid water use c=4.18 J/(g K) and neglect heat losses. Standard pressure is 1 bar.
For ionic activity use the 1 mol/kg molality standard state."""
ANSWER_FORMAT = """Required final line: one plain-text JSON object with exactly two keys.
Write "answer" as a JSON number in the target unit, and "unit" as exactly {unit}.
Start the line with an opening curly brace and end it with a closing curly brace.
The final JSON object must be outside Markdown, LaTeX, math boxes, and code fences."""
ARMS = {
    "direct": "Give only the required final answer JSON object. Do not show intermediate steps.",
    "cot": "Begin your response with 'Step 1:' and show two to four short numbered steps, including the equation you use and your intermediate calculations. Keep those steps visible in the response. After the steps, finish with the required plain-text JSON object.",
}
GENERATION = {
    "n_predict": 768,
    "temperature": 0.0,
    "top_k": 1,
    "seed": SEED,
    "cache_prompt": False,
    "return_tokens": True,
    "stop": ["<|im_end|>", "<|endoftext|>"],
}


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def questions_for(phase):
    source = ROOT / "data" / "expanded_100.jsonl"
    if sha256(source) != DATA_HASH or sha256(ROOT / "scoring.py") != SCORER_HASH:
        raise ValueError("SciBench data or scoring code differs from the pinned Local1 archive")
    formal = [q for q in read_jsonl(source) if q["variant"] == "original"]
    if len(formal) != 100 or len({q["id"] for q in formal}) != 100:
        raise ValueError("Expected 100 original SciBench questions with distinct IDs")
    if phase == "formal":
        return formal
    pilot = [q for q in read_jsonl(ROOT / "data" / "curated.jsonl") if q["id"] in PILOT_IDS]
    if len(pilot) != len(PILOT_IDS) or {q["family_id"] for q in pilot} & {q["family_id"] for q in formal}:
        raise ValueError("Pilot questions must come from the held-out development families")
    return pilot


def prompt_for(q, arm):
    unit = json.dumps(q["target_unit"], ensure_ascii=False)
    user = f"{COMMON}\n\nProblem:\n{q['question']}\n\n{ANSWER_FORMAT.format(unit=unit)}\n{ARMS[arm]}"
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def request(port, endpoint, body=None, timeout=180):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(f"http://127.0.0.1:{port}/{endpoint}", data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def launch_args(executable, model, device, gpu_layers, port):
    try:
        model_arg = os.path.relpath(model, executable.parent)
    except ValueError:
        model_arg = str(model)  # Windows permits weights on a different drive.
    return [
        str(executable), "--model", model_arg, "--host", "127.0.0.1", "--port", str(port),
        "--parallel", "1", "--ctx-size", str(CONTEXT), "--threads", str(THREADS),
        "--n-gpu-layers", str(gpu_layers), "--device", device,
    ]


def wait_until_ready(port, proc):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}; see server.log")
        try:
            if request(port, "health", timeout=2)["status"] == "ok":
                return
        except (URLError, TimeoutError):
            time.sleep(0.25)
    raise RuntimeError("llama-server failed to become ready; see server.log")


def jobs(questions):
    rng = random.Random(SEED)
    for question in questions:
        arms = list(ARMS)
        rng.shuffle(arms)
        for arm in arms:
            yield question, arm


def generate(port, question, arm):
    prompt = prompt_for(question, arm)
    tokens = request(port, "tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    if not tokens or len(tokens) + GENERATION["n_predict"] > CONTEXT:
        raise ValueError(f"Invalid prompt length for {question['id']}")
    started = time.perf_counter()
    result = request(port, "completion", {**GENERATION, "prompt": tokens})
    seconds = time.perf_counter() - started
    output = result["content"]
    if result.get("truncated"):
        raise ValueError(f"Prompt truncated for {question['id']}")
    if not isinstance(output, str) or result.get("stop_type") not in {"eos", "word", "limit"}:
        raise ValueError(f"Unexpected completion response for {question['id']}")
    return {
        "id": question["id"],
        "family_id": question["family_id"],
        "source": question["source"],
        "domain": question["domain"],
        "arm": arm,
        "prompt": prompt,
        "output": output,
        "input_tokens": len(tokens),
        "output_tokens": result["tokens_predicted"],
        "stop_type": result["stop_type"],
        "seconds": round(seconds, 6),
        "timings": result.get("timings"),
        "score": score(output, question),
        "score_5pct": score(output, question, rtol=0.05),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "formal"])
    parser.add_argument("--server-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "_runtime/qwen3-1.7b-Q4_K_M.gguf")
    parser.add_argument("--device", default="Vulkan0")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = (args.out or ROOT / "results" / args.phase).resolve()
    out.mkdir(parents=True, exist_ok=True)
    questions = questions_for(args.phase)
    model = args.model.resolve(strict=True)
    executable = args.server_exe.resolve(strict=True)
    if sha256(model) != MODEL_HASH:
        raise ValueError("Model digest differs from the pinned Local1 model")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    version = subprocess.run([str(executable), "--version"], capture_output=True, text=True,
                             check=True, creationflags=creationflags)
    version_text = (version.stdout + version.stderr).strip()
    if "build 12404" not in version_text:
        raise ValueError(f"Expected llama.cpp b12404. Found: {version_text}")
    config = {
        "phase": args.phase,
        "source_commit": ARCHIVE_COMMIT,
        "expanded_data_sha256": DATA_HASH,
        "scorer_sha256": SCORER_HASH,
        "runner_sha256": sha256(Path(__file__)),
        "model": "Qwen3-1.7B Q4_K_M",
        "model_sha256": MODEL_HASH,
        "backend_version": version_text,
        "backend_sha256": sha256(executable),
        "platform": platform.platform(),
        "device": args.device,
        "gpu_layers": args.gpu_layers,
        "threads": THREADS,
        "context": CONTEXT,
        "thinking": False,
        "generation": GENERATION,
        "question_ids": [q["id"] for q in questions],
    }
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Run settings changed. Choose a new output directory.")
    else:
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record_path = out / "records.jsonl"
    expected = {(q["id"], arm) for q in questions for arm in ARMS}
    previous = read_jsonl(record_path) if record_path.exists() else []
    done = {(r["id"], r["arm"]) for r in previous}
    if len(done) != len(previous) or not done <= expected:
        raise ValueError("Existing records contain duplicate or unexpected jobs")
    if done == expected:
        print(f"All {len(expected)} completions already present in {out}")
        return

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (out / "server.log").open("ab") as log, record_path.open("a", encoding="utf-8") as records:
        proc = subprocess.Popen(launch_args(executable, model, args.device, args.gpu_layers, port),
                                stdout=log, stderr=log, cwd=executable.parent, creationflags=creationflags)
        try:
            wait_until_ready(port, proc)
            for q, arm in jobs(questions):
                key = (q["id"], arm)
                if key in done:
                    continue
                row = generate(port, q, arm)
                records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                records.flush()
                done.add(key)
                print(f"{len(done)}/{len(expected)} {q['id']} {arm} correct={row['score']['correct']} "
                      f"tokens={row['output_tokens']} {row['seconds']:.2f}s", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()

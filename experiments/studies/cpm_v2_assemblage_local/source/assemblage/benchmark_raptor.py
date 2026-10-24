#!/usr/bin/env python3

import json
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Configuration
RAPTOR_SCRIPT = Path(__file__).with_name(
    "raptor.py"
)  # assumes raptor.py is next to this file
ITERATIONS_PER_MODEL = 30
MODELS = [
    "church",
    "celebahq",
    "bedroom",
    "cat",
]
WORDS_PER_ITERATION = 300


def build_cmd(model: str, iteration: int) -> list[str]:
    return [
        sys.executable,  # use the same Python interpreter that runs this script
        str(RAPTOR_SCRIPT),
        model,
        str(iteration),
        str(WORDS_PER_ITERATION),
    ]


# Run the experiments and gather the JSON objects
collected: list[object] = []

for model in MODELS:
    for i in tqdm(
        range(1, ITERATIONS_PER_MODEL + 1),
        desc=f"Model {model}",
        unit="run",
        leave=True,  # leave the bar on screen after finishing the model
    ):
        cmd = build_cmd(model, i)
        try:
            # Capture stdout; raise on non‑zero exit status
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            # If a single run fails, we log and continue
            sys.stderr.write(
                f"[ERROR] raptor.py failed for model={model}, iter={i}\n"
                f"Exit code: {exc.returncode}\n"
                f"Stdout: {exc.stdout}\n"
                f"Stderr: {exc.stderr}\n"
            )
            continue
        # Parse the JSON that raptor.py printed to stdout
        try:
            obj = json.loads(completed.stdout)
        except json.JSONDecodeError as jde:
            sys.stderr.write(
                f"[ERROR] Invalid JSON from raptor.py (model={model}, iter={i})\n"
                f"Output was: {completed.stdout!r}\n"
                f"Exception: {jde}\n"
            )
            continue

        # Enrich the object with provenance info
        if isinstance(obj, dict):
            obj.setdefault("model", model)
            obj.setdefault("iteration", i)

        collected.append(obj)

# Write the aggregated data to a timestamped JSON file
timestamp = f"{int(time.time())}"
out_path = Path(f"benchmarks/raptor_{timestamp}.json")

try:
    out_path.write_text(json.dumps(collected, indent=2))
    print(f"[INFO] Saved {len(collected)} results to {out_path}")
except OSError as e:
    sys.stderr.write(f"[ERROR] Could not write results file: {e}\n")
    sys.exit(1)

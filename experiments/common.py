"""Shared experiment execution and explicit run metadata."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.problem import load_problem
from src.solver import solve


def run_jobs(input_path, jobs, output):
    problem = load_problem(input_path)
    digest = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    records = []
    for method, seed, config, sensitivity in jobs:
        result = solve(problem, method, seed, config, sensitivity=sensitivity)
        row = {
            key: value
            for key, value in result.items()
            if key not in {"sequence", "direction", "config"}
        }
        row.update(result["config"])
        row["sequence"] = json.dumps(result["sequence"])
        row["direction"] = json.dumps(result["direction"])
        row["input_sha256"] = digest
        records.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Saved {len(records)} runs: {output}")
    return records

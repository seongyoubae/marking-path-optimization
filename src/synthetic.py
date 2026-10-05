"""Generate paired endpoints without using industrial geometry."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.problem import Problem


def generate_problem(parts=8, seed=7):
    if parts < 2:
        raise ValueError("At least two parts are required")
    rng = np.random.default_rng(seed)
    centers = rng.uniform([12, 12], [88, 65], size=(parts, 2))
    angles = rng.uniform(0, np.pi, size=parts)
    half_lengths = rng.uniform(2, 6, size=parts)
    offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * half_lengths[:, None]
    coords = np.stack([centers - offsets, centers + offsets], axis=1).reshape(-1, 2)
    paired = coords[np.arange(2 * parts) ^ 1]
    return Problem(np.zeros(2), coords, paired, "synthetic_demo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/sample/synthetic_parts.json"))
    args = parser.parse_args()
    problem = generate_problem(args.parts, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(problem.to_dict(), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

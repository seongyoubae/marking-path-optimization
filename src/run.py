"""Run an optimizer on a public JSON instance."""

import argparse
import json
from pathlib import Path

from src.problem import load_problem
from src.solver import METHODS, RunConfig, solve


def add_run_arguments(parser):
    parser.add_argument("--input", type=Path, default=Path("data/sample/synthetic_parts.json"))
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument(
        "--initialize", choices=["nearest neighbor", "random"], default="nearest neighbor"
    )
    parser.add_argument("--coupling", type=float, default=0.05)
    parser.add_argument("--sa-initial", type=float, default=1.0)
    parser.add_argument("--sa-minimum", type=float, default=0.2)
    parser.add_argument("--sa-cooling", type=float, default=0.6)
    parser.add_argument("--time-limit", type=float, default=30.0)


def config_from_args(args):
    return RunConfig(
        args.population,
        args.generations,
        args.initialize,
        args.coupling,
        args.sa_initial,
        args.sa_minimum,
        args.sa_cooling,
        args.time_limit,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--method", choices=METHODS, default="SAC-GWO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/solution.json"))
    args = parser.parse_args()
    result = solve(load_problem(args.input), args.method, args.seed, config_from_args(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"{result['method']}: normalized travel {result['travel_cost_normalized']:.6f} | {args.output}"
    )


if __name__ == "__main__":
    main()

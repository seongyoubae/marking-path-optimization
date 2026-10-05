"""Repeated seeded runs; solver budgets are recorded, not assumed equivalent."""

import argparse
from pathlib import Path

from experiments.common import run_jobs
from src.run import add_run_arguments, config_from_args
from src.solver import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=["SAC-GWO", "GA", "PSO", "SA", "ACO"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.csv"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    config = config_from_args(args)
    jobs = [(method, seed, config, False) for method in args.methods for seed in args.seeds]
    run_jobs(args.input, jobs, args.output)


if __name__ == "__main__":
    main()

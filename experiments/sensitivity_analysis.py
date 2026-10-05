"""Coupling sweep using the separately uploaded sensitivity optimizer."""

import argparse
from dataclasses import replace
from pathlib import Path

from experiments.common import run_jobs
from src.run import add_run_arguments, config_from_args

SOURCE_COUPLING_GRID = [0.0, 0.025, 0.05, 0.075, 0.1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--values", nargs="+", type=float, default=SOURCE_COUPLING_GRID)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("outputs/sensitivity.csv"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.values)) != len(args.values):
        parser.error("Seeds and coupling values must be unique")
    config = config_from_args(args)
    jobs = [
        ("SAC-GWO", seed, replace(config, coupling=k), True)
        for k in args.values
        for seed in args.seeds
    ]
    run_jobs(args.input, jobs, args.output)


if __name__ == "__main__":
    main()

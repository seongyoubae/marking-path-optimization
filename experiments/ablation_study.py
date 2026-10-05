"""Run the five flag configurations defined in the uploaded SAC-GWO source."""

import argparse
from pathlib import Path

from experiments.common import run_jobs
from src.run import add_run_arguments, config_from_args
from src.solver import ABLATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation.csv"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    config = config_from_args(args)
    run_jobs(
        args.input,
        [(method, seed, config, False) for method in ABLATIONS for seed in args.seeds],
        args.output,
    )


if __name__ == "__main__":
    main()

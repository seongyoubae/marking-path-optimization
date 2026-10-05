"""Two-sided Wilcoxon rank-sum comparisons, following scipy.stats.ranksums in the source."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ranksums


def compare_groups(frame, reference="SAC-GWO", alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    required = {"method", "travel_cost_normalized"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    # Avoid pooling different instances, source forks, or parameter settings.
    for column in [
        "input_sha256",
        "instance",
        "implementation",
        "coupling",
        "population",
        "generations",
        "initialize",
        "sa_initial",
        "sa_minimum",
        "sa_cooling",
        "time_limit",
    ]:
        if column in frame and frame[column].nunique(dropna=False) > 1:
            raise ValueError(f"Select one {column} setting before statistical comparison")
    if "seed" in frame and frame.duplicated(["method", "seed"]).any():
        raise ValueError("Duplicate method/seed observations")
    values = pd.to_numeric(frame["travel_cost_normalized"], errors="raise")
    if np.isinf(values).any():
        raise ValueError("Infinite costs are not valid observations")
    groups = {
        method: values[frame["method"] == method].dropna().to_numpy()
        for method in frame["method"].unique()
    }
    if reference not in groups or len(groups[reference]) < 2:
        raise ValueError("Reference needs at least two observations")
    rows = []
    for method, sample in groups.items():
        if method == reference:
            continue
        if len(sample) < 2:
            raise ValueError(f"{method} needs at least two observations")
        statistic, pvalue = ranksums(groups[reference], sample)
        rows.append(
            {
                "reference": reference,
                "comparison": method,
                "n_reference": len(groups[reference]),
                "n_comparison": len(sample),
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "alpha": alpha,
                "reject_at_alpha": bool(pvalue < alpha),
                "alternative": "two-sided",
                "p_value_adjustment": "none",
            }
        )
    if not rows:
        raise ValueError("At least two methods are required")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/benchmark.csv"))
    parser.add_argument("--reference", default="SAC-GWO")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("outputs/rank_sum.csv"))
    args = parser.parse_args()
    result = compare_groups(pd.read_csv(args.input), args.reference, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved unadjusted rank-sum comparisons: {args.output}")


if __name__ == "__main__":
    main()

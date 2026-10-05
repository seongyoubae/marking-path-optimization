"""Plot processing strokes and the travel legs actually scored by the objective."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.problem import load_problem, travel_cost, validate_solution


def plot_route(problem, solution, output, title="Synthetic route example"):
    sequence, direction = validate_solution(problem, solution)
    idx = 2 * sequence + direction
    entries, exits = problem.coords[idx], problem.coords_paired[idx]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#fafbfe")
    ax.set_facecolor("#fafbfe")
    previous = problem.origin
    for rank, (part, entry, exit_) in enumerate(zip(sequence, entries, exits), 1):
        ax.annotate(
            "",
            xy=entry,
            xytext=previous,
            arrowprops=dict(
                arrowstyle="->", color="#8495b5", lw=1.5, linestyle="--", shrinkA=2, shrinkB=4
            ),
        )
        ax.annotate(
            "",
            xy=exit_,
            xytext=entry,
            arrowprops=dict(arrowstyle="->", color="#006c78", lw=3, shrinkA=0, shrinkB=0),
        )
        middle = (entry + exit_) / 2
        ax.annotate(
            f"{rank}: P{part}",
            xy=middle,
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color="#153343",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fafbfe", ec="none", alpha=0.95),
        )
        ax.scatter(*entry, color="#006c78", s=28, zorder=4)
        previous = exit_
    ax.annotate(
        "",
        xy=problem.origin,
        xytext=previous,
        arrowprops=dict(
            arrowstyle="->", color="#8495b5", lw=1.5, linestyle="--", shrinkA=2, shrinkB=4
        ),
    )
    ax.scatter(*problem.origin, marker="s", color="#ea9150", s=65, zorder=5)
    ax.annotate(
        "Origin", problem.origin, xytext=(10, -4), textcoords="offset points", color="#8c4b21"
    )
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=22, color="#153343")
    ax.set_xlabel("x (synthetic coordinate units)")
    ax.set_ylabel("y (synthetic coordinate units)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.15)
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#006c78", lw=3, label="Processing stroke (fixed)"),
            Line2D([0], [0], color="#8495b5", linestyle="--", label="Travel (objective)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.07,
        0.02,
        "Synthetic paired endpoints · Each part is visited once · Direction selects entry and exit",
        fontsize=10,
        color="#546578",
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.23)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample/synthetic_parts.json"))
    parser.add_argument("--solution", type=Path, default=Path("outputs/solution.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/route.png"))
    args = parser.parse_args()
    problem = load_problem(args.input)
    result = json.loads(args.solution.read_text())
    solution = result["sequence"] + result["direction"]
    if not np.isclose(
        travel_cost(problem, solution, normalized=True), result["travel_cost_normalized"]
    ):
        raise ValueError("The solution file does not match the input geometry")
    plot_route(
        problem, solution, args.output, title=f"{result['method']} | Synthetic route example"
    )
    print(args.output)


if __name__ == "__main__":
    main()

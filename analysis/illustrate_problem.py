"""Regenerate the README schematic from deliberately simple synthetic endpoints."""

from analysis.plot_route import plot_route
from src.problem import Problem


def main():
    coords = [
        [18, 18],
        [29, 18],
        [44, 18],
        [55, 18],
        [70, 18],
        [81, 18],
        [70, 47],
        [81, 47],
        [44, 47],
        [55, 47],
        [18, 47],
        [29, 47],
    ]
    problem = Problem([0, 0], coords, [coords[i ^ 1] for i in range(12)], "synthetic_illustration")
    # A manually selected route, not an optimization result.
    solution = [0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 1, 1]
    plot_route(problem, solution, "assets/problem.png", title="Marking route: sequence + direction")


if __name__ == "__main__":
    main()

"""Public paired-endpoint input and independent route validation."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Problem:
    origin: np.ndarray
    coords: np.ndarray
    coords_paired: np.ndarray
    name: str = "sample"

    def __post_init__(self):
        origin = np.asarray(self.origin, dtype=float)
        coords = np.asarray(self.coords, dtype=float)
        paired = np.asarray(self.coords_paired, dtype=float)
        if origin.shape != (2,) or not np.array_equal(origin, [0, 0]):
            raise ValueError(
                "Public inputs require origin [0, 0] for consistent source normalization"
            )
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 4 or len(coords) % 2:
            raise ValueError("coords must have shape (2*n, 2), with at least two parts")
        if paired.shape != coords.shape:
            raise ValueError("coords_paired must have the same shape as coords")
        if not all(np.isfinite(a).all() for a in [origin, coords, paired]):
            raise ValueError("Coordinates must be finite")
        if (coords < 0).any() or coords.max() <= 0:
            raise ValueError("Coordinates must be nonnegative with a positive maximum")
        if not np.array_equal(paired, coords[np.arange(len(coords)) ^ 1]):
            raise ValueError("Each adjacent pair must be reversed in coords_paired")
        # The source SA nearest-neighbor initializer masks every zero distance.
        if len(np.unique(np.vstack([origin, coords]), axis=0)) != len(coords) + 1:
            raise ValueError("Endpoints must be distinct and must not coincide with the origin")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "coords_paired", paired)

    @property
    def num_parts(self):
        return len(self.coords) // 2

    @property
    def scale(self):
        return float(self.coords.max())

    def as_tuple(self):
        return self.origin.copy(), self.coords.copy(), self.coords_paired.copy()

    def to_dict(self):
        return {
            "name": self.name,
            "origin": self.origin.tolist(),
            "coords": self.coords.tolist(),
            "coords_paired": self.coords_paired.tolist(),
        }


def load_problem(path):
    data = json.loads(Path(path).read_text())
    return Problem(
        data["origin"], data["coords"], data["coords_paired"], data.get("name", "sample")
    )


def validate_solution(problem, solution):
    a = np.asarray(solution)
    n = problem.num_parts
    if a.shape != (2 * n,) or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError("Solution must contain 2*n integer values")
    sequence, direction = a[:n].astype(int), a[n:].astype(int)
    if not np.array_equal(np.sort(sequence), np.arange(n)):
        raise ValueError("Sequence must visit each part exactly once")
    if not np.isin(direction, [0, 1]).all():
        raise ValueError("Direction must contain only 0 or 1")
    return sequence, direction


def travel_cost(problem, solution, normalized=False):
    sequence, direction = validate_solution(problem, solution)
    idx = 2 * sequence + direction
    entries, exits = problem.coords[idx], problem.coords_paired[idx]
    previous = np.vstack([problem.origin, exits])
    following = np.vstack([entries, problem.origin])
    cost = float(np.linalg.norm(previous - following, axis=1).sum())
    return cost / problem.scale if normalized else cost

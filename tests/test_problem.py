import numpy as np
import pytest

from src.problem import Problem, travel_cost, validate_solution
from src.synthetic import generate_problem


@pytest.fixture
def two_parts():
    coords = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    return Problem([0, 0], coords, coords[[1, 0, 3, 2]])


def test_closed_travel_excludes_processing(two_parts):
    # 0->1, 2->3, 4->0: 1 + 1 + 4; processing adds a fixed 2.
    assert travel_cost(two_parts, [0, 1, 0, 0]) == 6
    assert travel_cost(two_parts, [0, 1, 0, 0], normalized=True) == 1.5


def test_direction_is_per_visit_position(two_parts):
    # Visit P1 backwards, then P0 forwards: 0->4, 3->1, 2->0.
    assert travel_cost(two_parts, [1, 0, 1, 0]) == 8


@pytest.mark.parametrize("solution", [[0, 0, 0, 1], [0, 1, 2, 0], [0, 1, 0], [0, 1.5, 0, 0]])
def test_invalid_solutions(two_parts, solution):
    with pytest.raises(ValueError):
        validate_solution(two_parts, solution)


@pytest.mark.parametrize("change", ["origin", "pair", "duplicate", "nan", "negative", "one_part"])
def test_public_input_constraints(change):
    data = generate_problem().to_dict()
    if change == "origin":
        data["origin"] = [1, 1]
    elif change == "pair":
        data["coords_paired"][0] = [9, 9]
    elif change == "duplicate":
        data["coords"][0] = data["coords"][1]
        data["coords_paired"][1] = data["coords"][0]
    elif change == "nan":
        data["coords"][0][0] = float("nan")
    elif change == "negative":
        data["coords"][0][0] = -1
    else:
        data["coords"] = data["coords"][:2]
        data["coords_paired"] = data["coords_paired"][:2]
    with pytest.raises(ValueError):
        Problem(**data)


def test_synthetic_generation_is_reproducible():
    assert generate_problem(8, 7).to_dict() == generate_problem(8, 7).to_dict()

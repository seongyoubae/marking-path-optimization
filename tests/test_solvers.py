import importlib
import itertools

import numpy as np
import pytest

from src.problem import travel_cost
from src.solver import ABLATIONS, RunConfig, solve
from src.synthetic import generate_problem


@pytest.mark.parametrize(
    "module",
    [
        "src.sac_gwo",
        "baselines.ga",
        "baselines.pso",
        "baselines.sa",
        "baselines.aco",
        "experiments.sensitivity_optimizer",
        "analysis.statistical_test",
    ],
)
def test_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("method", [*ABLATIONS, "GA", "PSO", "SA", "ACO"])
@pytest.mark.parametrize("parts", [2, 8])
def test_source_solvers_return_audited_routes(method, parts):
    problem = generate_problem(parts)
    result = solve(problem, method, 2, RunConfig(population=8, generations=5))
    route = result["sequence"] + result["direction"]
    assert result["travel_cost_raw"] == pytest.approx(travel_cost(problem, route))
    assert result["travel_cost_raw"] == pytest.approx(
        result["travel_cost_normalized"] * problem.scale
    )
    assert np.isfinite(result["native_cost"])


@pytest.mark.parametrize("method", ["SAC-GWO", "GA", "PSO", "SA", "ACO"])
def test_seeded_reproducibility(method):
    config = RunConfig(population=8, generations=8)
    results = [solve(generate_problem(), method, 3, config) for _ in range(2)]
    for field in ["sequence", "direction", "native_cost", "native_initial_cost"]:
        assert results[0][field] == results[1][field]


@pytest.mark.parametrize("coupling", [0.0, 0.025, 0.05, 0.075, 0.1])
def test_distinct_sensitivity_source(coupling):
    result = solve(
        generate_problem(),
        "SAC-GWO",
        4,
        RunConfig(population=8, generations=5, coupling=coupling),
        sensitivity=True,
    )
    assert result["implementation"] == "source_sensitivity"


@pytest.mark.gurobi
def test_optional_gurobi_matches_exhaustive_small_instance():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        env.dispose()
    except gp.GurobiError:
        pytest.skip("No working Gurobi license")
    problem = generate_problem(3)
    exhaustive = min(
        travel_cost(problem, list(seq) + list(dirs), normalized=True)
        for seq in itertools.permutations(range(3))
        for dirs in itertools.product([0, 1], repeat=3)
    )
    result = solve(problem, "Gurobi", config=RunConfig(time_limit=10))
    assert result["status"] == "Optimal"
    assert result["mip_gap_percent"] == pytest.approx(0)
    assert result["travel_cost_normalized"] == pytest.approx(exhaustive, rel=1e-5)

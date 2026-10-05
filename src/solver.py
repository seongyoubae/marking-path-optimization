"""Seeded adapters around the uploaded optimizer classes."""

import contextlib
import io
import random
from dataclasses import asdict, dataclass

import numpy as np

from src.problem import travel_cost, validate_solution
from src.sac_gwo import GreyWolfOptimizerNesting

ABLATIONS = {
    "GWO": dict(adaptive_a=False, chaos_for_position_update=None, advanced_gwo=False),
    "C-GWO": dict(adaptive_a=False, chaos_for_position_update="logistic", advanced_gwo=False),
    "A-GWO": dict(adaptive_a=True, chaos_for_position_update=None, advanced_gwo=False),
    "AC-GWO": dict(adaptive_a=True, chaos_for_position_update="logistic", advanced_gwo=False),
    "SAC-GWO": dict(adaptive_a=True, chaos_for_position_update="logistic", advanced_gwo=True),
}
METHODS = (*ABLATIONS, "GA", "PSO", "SA", "ACO", "Gurobi")


@dataclass(frozen=True)
class RunConfig:
    population: int = 30
    generations: int = 100
    initialize: str = "nearest neighbor"
    coupling: float = 0.05
    sa_initial: float = 1.0
    sa_minimum: float = 0.2
    sa_cooling: float = 0.6
    time_limit: float = 30.0

    def __post_init__(self):
        if not np.isfinite(
            [self.coupling, self.sa_initial, self.sa_minimum, self.sa_cooling, self.time_limit]
        ).all():
            raise ValueError("Solver control parameters must be finite")
        if self.population < 5 or self.generations < 1:
            raise ValueError("population >= 5 and generations >= 1 are required")
        if self.initialize not in {"nearest neighbor", "random"}:
            raise ValueError("Unsupported initialization")
        if not 0 <= self.coupling <= 0.5:
            raise ValueError("coupling must be in [0, 0.5], keeping tau <= 1")
        if not (self.sa_initial > self.sa_minimum > 0 and 0 < self.sa_cooling < 1):
            raise ValueError("Require SA initial > minimum > 0 and 0 < cooling < 1")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")


def solve(problem, method="SAC-GWO", seed=42, config=None, sensitivity=False):
    config = config or RunConfig()
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if sensitivity and method != "SAC-GWO":
        raise ValueError("The source sensitivity implementation is used only for SAC-GWO sweeps")
    random.seed(seed)
    np.random.seed(seed)
    data = problem.as_tuple()
    status, gap = "completed", None
    with contextlib.redirect_stdout(io.StringIO()):
        if method in ABLATIONS:
            optimizer_class = GreyWolfOptimizerNesting
            if sensitivity:
                from experiments.sensitivity_optimizer import (
                    GreyWolfOptimizerNesting as SensitivityOptimizer,
                )

                optimizer_class = SensitivityOptimizer
            optimizer = optimizer_class(
                population_size=config.population,
                generations=config.generations,
                initialize_method=config.initialize,
                **ABLATIONS[method],
            )
            optimizer.k_coupling = config.coupling
            solution, native_cost, initial, duration = optimizer.run_gwo_for_instance(data)
        elif method == "GA":
            from baselines.ga import GeneticAlgorithmNestingSAStyle

            optimizer = GeneticAlgorithmNestingSAStyle(
                population_size=config.population,
                generations=config.generations,
                initialize_method=config.initialize,
            )
            solution, native_cost, initial, duration = optimizer.run_ga_for_instance(data)
        elif method == "PSO":
            from baselines.pso import LPSPOOptimizer

            optimizer = LPSPOOptimizer(
                config.population,
                config.generations,
                use_nn_init=config.initialize == "nearest neighbor",
            )
            native_cost, duration, initial = optimizer.run(data)
            position = optimizer.gbest_position
            n = problem.num_parts
            solution = np.concatenate([np.argsort(position[:n]), (position[n:] > 0.5).astype(int)])
        elif method == "SA":
            from baselines.sa import SimulatedAnnealing

            optimizer = SimulatedAnnealing(
                data,
                config.sa_initial,
                config.sa_minimum,
                config.sa_cooling,
                initialize=config.initialize,
            )
            initial = optimizer.objective_init
            solution, native_cost, duration = optimizer.run()
        elif method == "ACO":
            from baselines.aco import AntColonyOptimizerNesting

            optimizer = AntColonyOptimizerNesting(
                population_size=config.population, generations=config.generations
            )
            solution, native_cost, initial, duration = optimizer.run_aco_for_instance(data)
        else:
            try:
                from baselines.gurobi import GurobiOptimizerFull
            except ImportError as exc:
                raise RuntimeError(
                    "Install requirements_gurobi.txt to run the optional Gurobi baseline"
                ) from exc
            optimizer = GurobiOptimizerFull(time_limit=config.time_limit)
            solution, native_cost, gap, status, duration = optimizer.run_gurobi_for_instance(data)
            initial = None
    if solution is None or not np.isfinite(native_cost):
        raise RuntimeError(f"{method} returned no feasible solution (status: {status})")
    sequence, direction = validate_solution(problem, solution)
    normalized = travel_cost(problem, solution, normalized=True)
    raw = travel_cost(problem, solution)
    # ACO evaluates raw input coordinates; the other sources normalize internally.
    expected = raw if method == "ACO" else normalized
    if not np.isclose(native_cost, expected, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"{method} objective and returned route disagree")
    return {
        "method": method,
        "implementation": "source_sensitivity" if sensitivity else "main",
        "seed": seed,
        "instance": problem.name,
        "parts": problem.num_parts,
        "sequence": sequence.tolist(),
        "direction": direction.tolist(),
        "travel_cost_raw": raw,
        "travel_cost_normalized": normalized,
        "native_cost": float(native_cost),
        "native_initial_cost": None if initial is None else float(initial),
        "native_cost_units": "input" if method == "ACO" else "normalized",
        "runtime_seconds": float(duration),
        "status": status,
        "mip_gap_percent": gap,
        "config": asdict(config),
    }

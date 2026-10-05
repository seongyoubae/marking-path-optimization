# Implementation and provenance notes

## Retained source implementations

| Uploaded source | Public module | Treatment |
| :--- | :--- | :--- |
| `sac-gwo(1).py` | `src/sac_gwo.py` | Main optimizer class retained; private batch runner replaced. |
| `ga(1).py` | `baselines/ga.py` | GA class retained. |
| `PSO(1).py` | `baselines/pso.py` | Base class and LPSPSO variant retained, including the original class spelling. |
| `sa(5).py` | `baselines/sa.py` | SA class retained; progress output suppressed by the adapter. |
| `ACO(4).py`, `aco3(1).py` | `baselines/aco.py` | Class definitions are identical; one representative is retained. |
| `gurobi(4).py` | `baselines/gurobi.py` | Optimizer class retained; license environment mutation and private runner removed. Console messages translated. |
| Sensitivity analysis upload | `experiments/sensitivity_optimizer.py` | Distinct optimizer retained separately; public sweep uses its original coupling grid. |
| `Wilcoxon(1).py` | `analysis/statistical_test.py` | Same `scipy.stats.ranksums` test, moved from a top-level private Excel script into a CSV function/CLI. |

Comments and docstrings were cleaned, imports reduced to algorithm dependencies, and formatting standardized. No search equations, operators, acceptance rules or population-update statements were changed in the retained classes. Public adapters handle validation, RNG seeds, solver selection, result conversion and paths separately.

## Main and sensitivity GWO are different

The main implementation excludes the actual alpha/beta/delta indices when updating non-leader wolves. The uploaded sensitivity implementation instead updates positions `3` through `population_size-1`, after copying leaders. Their crossover/control code also differs: the sensitivity implementation calls leader perturbation directly in its leader-following branch, while the main implementation conditions that call on the chaos configuration outside its advanced branch.

Both are retained to avoid changing the uploaded sensitivity study into an experiment on another optimizer. Benchmark and ablation use `main`; the coupling sweep uses `source_sensitivity`. These labels are saved in every experiment row. A difference between their outcomes must not be attributed solely to coupling.

The main source lists all five ablation configurations, although its original executable block enables only C-GWO. The public runner deliberately selects all three SAC-GWO flags for `--method SAC-GWO`; this is a configuration selection, not evidence of an original SAC-GWO result.

## Objective and input boundaries

All retained evaluators sum origin-to-entry, exit-to-next-entry and final-exit-to-origin distances. Processing strokes are not in that sum. Endpoint pairs in the public format are reversible, so processing length is constant across routes.

Main GWO, GA, SA and Gurobi normalize endpoints but leave the origin unchanged. PSO normalizes the origin too. ACO does not normalize. The public input contract fixes the origin to zero; the common evaluator recomputes raw and normalized travel and checks agreement with each source's native objective. A returned permutation/direction vector is also independently validated.

The source ACO deposits pheromone on edges between selected entry-node IDs, even though its objective transfers from the preceding exit to the next entry. That behavior is retained; it should not be described as an exact pheromone encoding of objective edges.

The main and sensitivity GWO implementations record best fitness before the generation update and do not reevaluate the last generated population before returning. This bookkeeping is retained.

Gurobi contains two origin direction indices, while the explicit origin departure/return constraints use index `0`. Its source status label maps every feasible non-optimal outcome to `TimeLimit`. The adapter validates the returned route and objective rather than assuming that a source status proves feasibility or optimality. It raises an error if extraction produces an incomplete route. The model formulation and status mapping have not been rewritten.

## Public experiment settings

| Setting | Public default |
| :--- | :--- |
| Population / generations | 30 / 100 |
| Initialization | Nearest neighbor where configurable; ACO always uses NN |
| Main GWO mutation | Source constructor defaults: sequence `0.02`, direction `0.01` |
| GA | Source constructor defaults, including elite size `5`, tournament size `5` |
| PSO variant | Source constructor defaults; NN initialization enabled by the adapter |
| SA | Initial temperature `1.0`, minimum `0.2`, cooling `0.6` |
| Coupling / diversity threshold | `0.05` / source threshold `0.05` |
| Gurobi time limit | 30 seconds |

These settings support a quick public example. The supplied main GWO, GA, PSO and sensitivity executable blocks use population `100` and generations `1000`; the supplied SA executable block uses temperature `450`, minimum `20`, cooling `0.95`. GA's original runner also overrides some constructor defaults. Private input availability, dataset composition and original experimental outputs were not established by these scripts. Hard-coded indexing or sample caps are not treated as dataset-size evidence.

Each repeated-run command seeds Python and NumPy independently for each solver/run and stores the full public configuration. Gurobi uses its own source defaults and a wall-clock limit, so strict cross-machine reproducibility is not asserted. Initial costs have method-specific meanings; they are retained as `native_initial_cost` and are not used as a common improvement metric.

## Verification scope

Tests cover the closed travel objective, direction semantics, input constraints, feasible routes for stochastic methods and GWO configurations, seeded reproducibility and statistical grouping. An optional small-instance Gurobi test compares its route cost with exhaustive enumeration when a working license is available. CI excludes the optional commercial solver.

The release review also compares the retained classes with the uploads after removing comments/docstrings and translating Gurobi console text. The public repository contains no private source runners or research datasets. Passing these checks establishes a runnable public example; it does not validate unprovided industrial experiments.

The marking schematic is reproducible with `python -m analysis.illustrate_problem`, which writes `assets/problem.svg`. Its four contours and eight marking operations are independently constructed synthetic geometry. Two larger plate shapes each contain three marking operations, while two bracket shapes contain one each. The example retains the paired-endpoint explanation established with the supplied schematic while using its own outlines and arrangement. These coordinates are illustrative, not production geometry or a dimensioned fabrication drawing. The source's `part` IDs index paired-endpoint route items; drawn contours are not a one-to-one catalog of those items. Dashed connectors show all nine travel legs, including approach and return. Operation IDs are displayed from 1, while the Python solution uses zero-based IDs. Direction bits follow visit positions, including the example's final operation 8 then operation 7. The manually selected route illustrates the model and is not a measured SAC-GWO result. Contours and openings do not add collision constraints or change the objective. Neutral material fills, dark outlines and white numbered badges distinguish the contours and operation IDs. Solid red marking strokes and dark dashed transfer arrows remain distinguishable by line style when viewed in grayscale.

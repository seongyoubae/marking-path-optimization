# Marking Path Optimization with SAC-GWO

**Choose the order. Choose the direction. Reduce the travel between marking operations.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Method](https://img.shields.io/badge/Method-SAC--GWO-006C78)
![Data](https://img.shields.io/badge/Data-Synthetic_example-64748B)

An engineering research project for marking routes in shipbuilding production: optimize how a tool visits parts on a fixed layout, using a discrete Grey Wolf Optimizer with state-aware, adaptive and chaotic control.

| At a glance | |
| :--- | :--- |
| **Problem** | Minimize travel from the origin, between parts, and back to the origin by choosing visit order and processing direction. |
| **Proposed method** | SAC-GWO: a Grey Wolf Optimizer variant with population-state feedback, adaptive search control, logistic chaos and coupled operator selection. |
| **Baselines** | GA, PSO (the uploaded LPSPSO variant), SA, ACO and optional Gurobi. |
| **What I implemented** | Sequence/direction search, paired-endpoint distance evaluation, solver comparisons, five GWO configurations, coupling sensitivity and rank-sum analysis. |
| **Key technical elements** | Permutation-safe operators, binary directions, fitness-diversity feedback, seeded experiments and independent route validation. |

## Overview

In a marking workflow, processing each part is only one portion of the tool's movement. The tool must also reach the next part, enter from an appropriate endpoint and eventually return to the origin. A fixed nesting layout can therefore produce different travel distances depending on the visit order and direction.

This project treats those two decisions as a joint optimization problem. The repository organizes the supplied research implementations into importable solvers and adds a public JSON interface, synthetic example, reproducible execution commands and route checks.

**SAC-GWO here is an improved Grey Wolf Optimizer.** The supplied implementation uses state-aware/adaptive/chaotic search control; it does not contain a Soft Actor-Critic agent, actor/critic networks or reinforcement-learning training.

## Problem Definition

Each part has two endpoints. The tool enters through one endpoint and finishes at its paired endpoint. A feasible route must:

- Visit every part exactly once.
- Select one of two processing directions for each visit.
- Start and finish at the specified origin.

The part coordinates are fixed inputs. The optimizer selects the route; it does not generate a nesting layout, move parts, enforce collision constraints or model machine acceleration.

## Solution Representation

For `n` parts, a discrete candidate contains `2*n` integers:

```text
sequence  = [2, 0, 1]  # a permutation of zero-based part IDs
direction = [1, 0, 1]  # one direction bit per position in the sequence
solution  = [2, 0, 1, 1, 0, 1]
```

At visit position `i`, the selected coordinate row is:

```python
row = 2 * sequence[i] + direction[i]
entry = coords[row]
exit_ = coords_paired[row]
```

Direction `0` traverses the first endpoint to the second; direction `1` reverses that traversal. Direction bits belong to **visit positions**, rather than a separate array indexed by part ID. The PSO implementation uses continuous random keys, decoded with `argsort` for sequence and a `0.5` threshold for directions.

## Proposed Method: SAC-GWO

The main implementation is in [`src/sac_gwo.py`](src/sac_gwo.py). The public runner enables `adaptive_a`, logistic chaos and `advanced_gwo` together for SAC-GWO.

Here $t$ is the generation index, $T$ is the number of generations, $f$ contains the population fitness values, and $\kappa$ is the `k_coupling` parameter.

| Mechanism | Behavior implemented in the source |
| :--- | :--- |
| Population state | Measures normalized fitness diversity as $D = \sigma(f)/\mu(f)$, using the population fitness values when the mean exceeds $10^{-9}$. |
| Adaptive control | Starts from $a_{\mathrm{base}}(t) = 2 - 2t/T$. Below the diversity threshold $0.05$, SAC-GWO increases $a$ proportionally, capped at $2.0$. |
| Chaotic perturbation | Updates a logistic state with $z \leftarrow 4z(1-z)$ and uses $2z > 1$ to trigger a swap in a leader's sequence. |
| Coupled control | Sets the advanced-operator probability to $\tau = \kappa a$; the source default coupling is $\kappa = 0.05$. |
| Operator selection | In the advanced branch, $\lvert A\rvert > 1$ selects block insertion; otherwise it selects segment inversion. |
| Leader guidance | Uses alpha, beta and delta candidates with selection weights `0.5`, `0.3` and `0.2`, permutation crossover and direction mixing. |
| Update acceptance | Accepts a candidate only when its evaluated travel cost improves; copies the three leaders into the next population. |

Sequence swap and direction-bit mutation are also available. Segment inversion reverses the sequence segment without reversing its direction bits; it should not be read as a full geometric 2-opt implementation. Block insertion is present in the base update as well as the advanced branch.

## Baseline Algorithms

| Baseline | Supplied implementation |
| :--- | :--- |
| **GA** | Tournament selection, elitism, position-based sequence crossover, uniform direction crossover, swap mutation and bit flips. |
| **PSO** | The supplied LPSPSO variant: random-key decoding, time-varying coefficients, Singer-map control and Lévy-flight updates. It is not a separate vanilla PSO implementation. |
| **SA** | Sequence swaps and a direction flip, Metropolis acceptance and geometric cooling. Each temperature level evaluates $n^3$ proposals. |
| **ACO** | Nearest-neighbor initial route, endpoint-based probabilistic construction, evaporation and cost-based pheromone deposits. |
| **Gurobi — optional** | Mixed-integer route model with direction-specific arcs, flow constraints and MTZ subtour constraints. Reports the source status label and percentage MIP gap. |

Gurobi is installed separately and may require a commercial or other eligible license. A time-limited feasible solution is not an optimality certificate; check the returned status and gap. No license file or license location is bundled or assigned by the project.

## Optimization Objective

Let $o$ be the origin and $e_i$, $x_i$ the selected entry and exit at visit position $i$. The objective implemented by the solvers is:

```math
\begin{aligned}
L ={}& \left\lVert o-e_1 \right\rVert_2 \\
&+ \sum_{i=1}^{n-1} \left\lVert x_i-e_{i+1} \right\rVert_2 \\
&+ \left\lVert x_n-o \right\rVert_2
\end{aligned}
```

This is **non-processing travel**: approach, transfer and return movement. The entry-to-exit processing strokes are excluded. For reversible paired endpoints, their total length is fixed, so adding that constant would not change which route minimizes total movement.

The source solvers use different distance scales: ACO evaluates input coordinates directly; the other implementations normalize endpoint coordinates by their maximum scalar coordinate. The public adapter independently reevaluates every returned route and reports both `travel_cost_raw` and `travel_cost_normalized`. Use the common normalized field for comparisons, rather than mixing native solver costs.

Public inputs require origin `[0, 0]`, nonnegative coordinates and distinct, reversible endpoint pairs. These constraints avoid the source implementations' different origin-normalization conventions and zero-distance behavior without modifying their search logic.

## Experimental Setup

Only synthetic input is included. Industrial geometry, private pickle datasets and original result workbooks are excluded. No research performance numbers, dataset sizes or improvement claims are reported here.

The public demo defaults to 30 individuals, 100 generations and nearest-neighbor initialization where supported. ACO always uses its source nearest-neighbor initializer. These are runnable demo settings, not a reconstruction of the original experiments. The source SA schedule is reduced for the demo; its temperature settings are separately configurable.

Each experiment records the seed, input hash, implementation, solver settings, route, common travel costs, native cost and runtime. Python and NumPy RNGs are seeded for stochastic solvers. Population/generation settings do not give SA or Gurobi an equivalent evaluation budget, and runtime comparisons depend on the machine and solver configuration.

Source-specific details and preserved differences are documented in [`docs/implementation_notes.md`](docs/implementation_notes.md).

## Ablation and Sensitivity Analysis

The ablation runner uses the five configurations listed in the uploaded main source:

| Configuration | Adaptive $a$ | Logistic chaos | Advanced coupled operators |
| :--- | :---: | :---: | :---: |
| GWO | — | — | — |
| C-GWO | — | ✓ | — |
| A-GWO | ✓ | — | — |
| AC-GWO | ✓ | ✓ | — |
| SAC-GWO | ✓ | ✓ | ✓ |

These are variants of the supplied discrete GWO implementation, rather than claims of reproducing a separate textbook implementation. A-GWO and AC-GWO set $a = 1.8$ in their low-diversity branch; SAC-GWO uses the proportional boost described above.

The sensitivity runner uses the uploaded grid `k_coupling = [0.0, 0.025, 0.05, 0.075, 0.1]`. It preserves the optimizer from the separate sensitivity file in `experiments/sensitivity_optimizer.py`, because its update logic differs from the main implementation. The output labels that implementation as `source_sensitivity`; the two implementations are not silently merged.

## Statistical Validation

[`analysis/statistical_test.py`](analysis/statistical_test.py) follows the supplied use of `scipy.stats.ranksums`: a two-sided Wilcoxon rank-sum test between algorithm groups, with an explicit significance threshold (default `0.05`). It is an independent-sample rank-sum test, not a paired signed-rank test.

The analysis expects repeated runs of the same input and configuration. It rejects mixed settings and duplicate method/seed observations, removes missing costs and reports sample counts, test statistics and unadjusted p-values. The source test does not apply a multiple-comparison correction. Reusing seed labels does not make this a paired analysis; tied values and small samples need care when interpreting the asymptotic test.

The commands below demonstrate the pipeline. Their synthetic outputs do not establish research significance or industrial performance.

## Repository Structure

```text
marking-path-optimization/
├── README.md
├── requirements.txt
├── requirements_dev.txt
├── requirements_gurobi.txt
├── .gitignore
├── src/
│   ├── sac_gwo.py                 # main research optimizer
│   ├── problem.py                 # input validation and common objective
│   ├── solver.py                  # seeded adapters and ablation flags
│   ├── run.py                     # single-run CLI
│   └── synthetic.py               # public example generator
├── baselines/
│   ├── ga.py
│   ├── pso.py
│   ├── sa.py
│   ├── aco.py
│   └── gurobi.py
├── experiments/
│   ├── benchmark.py
│   ├── ablation_study.py
│   ├── sensitivity_analysis.py
│   ├── sensitivity_optimizer.py   # distinct source implementation
│   └── common.py
├── analysis/
│   ├── statistical_test.py
│   └── plot_route.py
├── data/sample/synthetic_parts.json
├── assets/problem.svg
├── docs/implementation_notes.md
├── tests/
└── .github/workflows/tests.yml
```

## How to Run

Run all commands from the repository root. The public project is tested with Python 3.12.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m src.run --method SAC-GWO --seed 42
python -m analysis.plot_route
```

The default input is the included synthetic example. The commands write a validated solution to `outputs/solution.json` and a route plot to `outputs/route.png`.

```bash
# Compare the proposed method and available non-commercial baselines
python -m experiments.benchmark --seeds 0 1 2 3 4

# Five GWO configurations and the original coupling grid
python -m experiments.ablation_study --seeds 0 1 2
python -m experiments.sensitivity_analysis --seeds 0 1 2

# Rank-sum comparisons on repeated runs of one input
python -m analysis.statistical_test --input outputs/benchmark.csv

# Generate another synthetic input; the part count is a demo choice
python -m src.synthetic --parts 12 --seed 9 --output outputs/synthetic_parts.json
python -m src.run --input outputs/synthetic_parts.json --method GA

# Optional licensed baseline
python -m pip install -r requirements_gurobi.txt
python -m src.run --method Gurobi --time-limit 30

# Import, objective, solver and analysis checks
python -m pip install -r requirements_dev.txt
python -m pytest -q
```

To supply your own publicly shareable geometry, follow [`data/README.md`](data/README.md). Input and output paths are command-line arguments; there is no dependency on a personal folder structure. Core modules can be imported without reading data, writing results or importing Gurobi.

## Tech Stack

- **Python / NumPy:** discrete candidates, geometry, random-key representations and population updates.
- **SciPy:** Lévy-flight helper functions and rank-sum testing.
- **Pandas:** experiment tables and CSV analysis.
- **Matplotlib:** route visualization with scored travel separated from processing strokes.
- **Gurobi / gurobipy, optional:** mixed-integer baseline.
- **pytest / Ruff:** verification and source checks.

## Project Context

This portfolio is based on research code for shipbuilding production marking-path optimization. It shows the connection from an industrial routing problem to a discrete solution representation, metaheuristic control mechanisms and an experimental comparison workflow.

The research solver logic is preserved. The public engineering layer adds input validation, consistent result reporting, synthetic data, command-line execution and tests. Private research inputs and results are not part of this repository.

## Author

**Seongyou Bae**  
M.S. in Naval Architecture and Ocean Engineering

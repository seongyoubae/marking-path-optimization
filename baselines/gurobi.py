"""Source-preserved optimizer; public execution is provided by src.run."""

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class GurobiOptimizerFull:
    """Preserved direction-specific mixed-integer routing model."""

    def __init__(self, time_limit=1800):
        self.time_limit = time_limit
        self.origin = None
        self.coords = None
        self.coords_paired = None
        self.num_parts = 0

    def _process_input_data(self, data_tuple_raw):
        if data_tuple_raw is None or len(data_tuple_raw) < 3:
            return False
        origin_np = np.array(data_tuple_raw[0], dtype=np.float32)
        coords_raw_np = np.array(data_tuple_raw[1], dtype=np.float32)
        coords_paired_raw_np = np.array(data_tuple_raw[2], dtype=np.float32)
        if coords_raw_np.ndim != 2 or coords_raw_np.shape[1] != 2:
            return False
        if coords_raw_np.size > 0:
            length_max = np.max(coords_raw_np)
            if length_max == 0:
                length_max = 1.0
            coords_normalized = coords_raw_np / length_max
            coords_paired_normalized = coords_paired_raw_np / length_max
        else:
            coords_normalized = coords_raw_np
            coords_paired_normalized = coords_paired_raw_np
        self.coords = coords_normalized
        self.coords_paired = coords_paired_normalized
        self.origin = origin_np
        self.num_parts = int(self.coords.shape[0] // 2)
        return True

    def run_gurobi_for_instance(self, instance_data_tuple):
        if not self._process_input_data(instance_data_tuple):
            return (None, float("inf"), 1.0, "Error", 0.0)
        N = self.num_parts
        if N == 0:
            return (np.array([]), 0.0, 0.0, "Empty", 0.0)
        start_time = time.time()
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            model = gp.Model("MPO_Full", env=env)
        except gp.GurobiError as e:
            print(f"\nLicense initialization failed: {e}")
            return (None, float("inf"), 1.0, "LicenseError", 0.0)
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("MIPGap", 0.0)
        nodes_pos = {
            0: {0: {"e": self.origin, "x": self.origin}, 1: {"e": self.origin, "x": self.origin}}
        }
        for i in range(N):
            nodes_pos[i + 1] = {
                0: {"e": self.coords[2 * i], "x": self.coords_paired[2 * i]},
                1: {"e": self.coords[2 * i + 1], "x": self.coords_paired[2 * i + 1]},
            }
        x = {}
        for u in range(N + 1):
            for k in [0, 1]:
                for v in range(N + 1):
                    for l in [0, 1]:
                        if u == v:
                            continue
                        dist = np.linalg.norm(nodes_pos[u][k]["x"] - nodes_pos[v][l]["e"])
                        x[u, k, v, l] = model.addVar(vtype=GRB.BINARY, obj=dist)
        u_mtz = {i: model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=N + 1) for i in range(1, N + 1)}
        for j in range(1, N + 1):
            model.addConstr(
                gp.quicksum(
                    (x[i, k, j, l] for i in range(N + 1) for k in [0, 1] for l in [0, 1] if i != j)
                )
                == 1
            )
        for j in range(1, N + 1):
            for l in [0, 1]:
                in_flow = gp.quicksum(
                    (x[i, k, j, l] for i in range(N + 1) for k in [0, 1] if i != j)
                )
                out_flow = gp.quicksum(
                    (x[j, l, n_j, n_l] for n_j in range(N + 1) for n_l in [0, 1] if j != n_j)
                )
                model.addConstr(in_flow == out_flow)
        model.addConstr(gp.quicksum((x[0, 0, j, l] for j in range(1, N + 1) for l in [0, 1])) == 1)
        model.addConstr(gp.quicksum((x[i, k, 0, 0] for i in range(1, N + 1) for k in [0, 1])) == 1)
        BigM = N + 2
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i != j:
                    x_sum = gp.quicksum((x[i, k, j, l] for k in [0, 1] for l in [0, 1]))
                    model.addConstr(u_mtz[i] - u_mtz[j] + BigM * x_sum <= BigM - 1)
        model.optimize()
        duration = time.time() - start_time
        if model.SolCount > 0:
            best_fitness = model.objVal
            res_gap = model.MIPGap * 100
            res_status = "Optimal" if model.status == GRB.OPTIMAL else "TimeLimit"
            sequence, direction = ([], [])
            curr, curr_d = (0, 0)
            for _ in range(N):
                found = False
                for nxt in range(1, N + 1):
                    if nxt == curr:
                        continue
                    for nxt_d in [0, 1]:
                        if x[curr, curr_d, nxt, nxt_d].X > 0.5:
                            sequence.append(nxt - 1)
                            direction.append(nxt_d)
                            curr, curr_d = (nxt, nxt_d)
                            found = True
                            break
                    if found:
                        break
            sol_array = np.concatenate([np.array(sequence), np.array(direction)])
            print(f"Solved: [{res_status}] Cost: {best_fitness:.4f}, Gap: {res_gap:.2f}%")
            return (sol_array, best_fitness, res_gap, res_status, duration)
        else:
            return (None, float("inf"), 100.0, "No_Feasible_Solution", duration)

"""Source-preserved optimizer; public execution is provided by src.run."""

import time
import numpy as np


class SimulatedAnnealing(object):
    """Swap-and-direction neighborhood with geometric temperature reduction."""

    def __init__(self, data, T_init, T_min, z, initialize="random"):
        self.T_init = T_init
        self.T_min = T_min
        self.z = z
        self.initialize = initialize
        self.origin, self.coords, self.coords_paired = self._processing_data(data)
        self.num_parts = int(len(self.coords) / 2)
        self.T = T_init
        self.solution = self._initialize()
        self.objective = self._evaluate(self.solution)
        self.objective_init = self.objective
        self.solution_opt = np.copy(self.solution)
        self.objective_opt = self.objective

    def _processing_data(self, data):
        origin = np.array(data[0])
        coords = np.array(data[1])
        coords_paired = np.array(data[2])
        length_max = np.max(coords)
        coords = coords / length_max
        coords_paired = coords_paired / length_max
        return (origin, coords, coords_paired)

    def _initialize(self):
        if self.initialize == "random":
            sequence = np.random.permutation(self.num_parts)
            direction = np.random.choice(2, size=self.num_parts)
        elif self.initialize == "nearest neighbor":
            coords_with_origin = np.concatenate([self.origin[np.newaxis, ...], self.coords], axis=0)
            diff = coords_with_origin[:, np.newaxis, :] - coords_with_origin[np.newaxis, :, :]
            distance_matrix = np.linalg.norm(diff, ord=2, axis=-1)
            distance_matrix = np.where(distance_matrix != 0.0, distance_matrix, np.inf)
            distance_matrix[:, 0] = np.inf
            current_idx = 0
            sequence, direction = ([], [])
            for i in range(self.num_parts):
                idx = np.argmin(distance_matrix[current_idx])
                order, dir = ((idx - 1) // 2, (idx - 1) % 2)
                sequence.append(order)
                direction.append(dir)
                distance_matrix[:, idx] = np.inf
                if dir == 0:
                    distance_matrix[:, idx + 1] = np.inf
                    current_idx = idx + 1
                else:
                    distance_matrix[:, idx - 1] = np.inf
                    current_idx = idx - 1
            sequence = np.array(sequence)
            direction = np.array(direction)
        else:
            print("Invalid initialization")
        initial_solution = np.concatenate([sequence, direction], axis=-1)
        return initial_solution

    def _evaluate(self, solution):
        idx = 2 * solution[: self.num_parts] + solution[self.num_parts :]
        starts_points = self.coords[idx]
        starts_points = np.concatenate([starts_points, self.origin[np.newaxis, ...]], axis=0)
        end_points = self.coords_paired[idx]
        end_points = np.concatenate([self.origin[np.newaxis, ...], end_points], axis=0)
        cost = np.linalg.norm(end_points - starts_points, ord=2, axis=-1).sum()
        return cost

    def _perturbate(self, solution):
        solution = np.copy(solution)
        point1, point2 = np.random.choice(self.num_parts, size=2, replace=False)
        solution[point1], solution[point2] = (solution[point2], solution[point1])
        point3 = np.random.choice(self.num_parts, size=1)
        solution[self.num_parts + point3] = (2 - solution[self.num_parts + point3]) // 2
        return solution

    def run(self):
        start = time.time()
        while self.T > self.T_min:
            step = 0
            while step < self.num_parts**3:
                new_solution = self._perturbate(self.solution)
                new_objective = self._evaluate(new_solution)
                delta_E = new_objective - self.objective
                if delta_E <= 0:
                    self.solution = new_solution
                    self.objective = new_objective
                    if self.objective <= self.objective_opt:
                        self.solution_opt = new_solution
                        self.objective_opt = new_objective
                else:
                    P = np.exp(-delta_E / self.T)
                    u = np.random.rand()
                    if u < P:
                        self.solution = new_solution
                        self.objective = new_objective
                step += 1
            print(self.T)
            self.T = self.z * self.T
        duration = time.time() - start
        return (self.solution_opt, self.objective_opt, duration)

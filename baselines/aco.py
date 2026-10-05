"""Source-preserved optimizer; public execution is provided by src.run."""

import random
import time
import numpy as np


class AntColonyOptimizerNesting:
    """Endpoint-based ACO initialized with a nearest-neighbor route."""

    def __init__(
        self,
        population_size=50,
        generations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.5,
        q_val=100.0,
        mode="ant_cycle",
    ):
        self.n_ants = population_size
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q_val = q_val
        self.mode = mode
        self.pheromone = None
        self.visibility = None
        self.all_points_coords = None
        self.distances = None
        self.origin = None
        self.coords = None
        self.coords_paired = None
        self.num_parts = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float("inf")
        self.current_initial_solution = None
        self.current_initial_fitness = float("inf")

    def _process_input_data(self, data_tuple_raw):
        if data_tuple_raw is None or len(data_tuple_raw) < 3:
            self.origin, self.coords, self.coords_paired, self.num_parts = (
                None,
                np.array([]),
                np.array([]),
                0,
            )
            return
        self.origin = np.array(data_tuple_raw[0], dtype=np.float32)
        self.coords = np.array(data_tuple_raw[1], dtype=np.float32)
        self.coords_paired = np.array(data_tuple_raw[2], dtype=np.float32)
        self.num_parts = int(self.coords.shape[0] // 2)

    def _initialize_instance_specifics(self, problem_instance_data_tuple):
        self._process_input_data(problem_instance_data_tuple)
        if self.num_parts == 0:
            return False
        self.all_points_coords = np.vstack([self.origin, self.coords])
        self.distances = np.linalg.norm(
            self.all_points_coords[:, np.newaxis, :] - self.all_points_coords[np.newaxis, :, :],
            axis=-1,
        )
        self.visibility = 1 / (self.distances + 1e-10)
        np.fill_diagonal(self.visibility, 0)
        self.pheromone = np.full(
            (len(self.all_points_coords), len(self.all_points_coords)),
            1.0 / len(self.all_points_coords) ** 2,
        )
        self.best_solution_overall = None
        self.best_fitness_overall = float("inf")
        return True

    def _evaluate_solution(self, solution_array):
        if self.coords is None or self.coords_paired is None or self.origin is None:
            return float("inf")
        if self.num_parts == 0:
            return 0.0
        if solution_array is None or solution_array.size == 0:
            return float("inf")
        seq_part = solution_array[: self.num_parts].astype(int)
        dir_part = solution_array[self.num_parts :].astype(int)
        try:
            idx = 2 * seq_part + dir_part
            if np.any(idx >= self.coords.shape[0]) or np.any(idx < 0):
                return float("inf")
            coords_A_sides = self.coords[idx]
            coords_B_sides = self.coords_paired[idx]
        except IndexError:
            return float("inf")
        starts_points_for_path = np.concatenate(
            [coords_A_sides, self.origin[np.newaxis, ...]], axis=0
        )
        end_points_for_path = np.concatenate([self.origin[np.newaxis, ...], coords_B_sides], axis=0)
        cost = np.linalg.norm(end_points_for_path - starts_points_for_path, ord=2, axis=-1).sum()
        return cost

    def _create_individual_nn(self):
        if (
            self.num_parts == 0
            or self.coords is None
            or self.coords.size == 0
            or (self.origin is None)
        ):
            return self._create_individual_random()
        all_points = np.concatenate([self.origin[np.newaxis, :], self.coords], axis=0)
        dist_matrix = np.linalg.norm(
            all_points[:, np.newaxis, :] - all_points[np.newaxis, :, :], axis=-1
        )
        visited_parts = [False] * self.num_parts
        sequence_nn, direction_nn = ([], [])
        current_point_idx = 0
        for _ in range(self.num_parts):
            distances = dist_matrix[current_point_idx].copy()
            distances[current_point_idx] = np.inf
            distances[0] = np.inf
            for i in range(self.num_parts):
                if visited_parts[i]:
                    distances[2 * i + 1] = np.inf
                    distances[2 * i + 2] = np.inf
            next_point_idx = np.argmin(distances)
            part_idx = (next_point_idx - 1) // 2
            is_A_side = (next_point_idx - 1) % 2 == 0
            sequence_nn.append(part_idx)
            direction_nn.append(0 if is_A_side else 1)
            visited_parts[part_idx] = True
            current_point_idx = next_point_idx + (1 if is_A_side else -1)
        return np.concatenate([sequence_nn, direction_nn])

    def _create_individual_random(self):
        if self.num_parts == 0:
            return np.array([])
        sequence_random = list(range(self.num_parts))
        random.shuffle(sequence_random)
        direction_random = [random.randint(0, 1) for _ in range(self.num_parts)]
        return np.concatenate([sequence_random, direction_random])

    def _mutate_solution(self, solution, swap_prob=0.5, flip_prob=0.1):
        mutated_solution = np.copy(solution)
        if random.random() < swap_prob and self.num_parts > 1:
            sequence_part = mutated_solution[: self.num_parts]
            idx1, idx2 = random.sample(range(self.num_parts), 2)
            sequence_part[idx1], sequence_part[idx2] = (sequence_part[idx2], sequence_part[idx1])
        if random.random() < flip_prob:
            dir_part = mutated_solution[self.num_parts :]
            dir_idx = random.randint(0, self.num_parts - 1)
            dir_part[dir_idx] = 1 - dir_part[dir_idx]
        return mutated_solution

    def _construct_ant_solutions(self):
        all_solutions = []
        for _ in range(self.n_ants):
            current_point_idx = 0
            visited_parts = set()
            sequence, direction = ([], [])
            while len(visited_parts) < self.num_parts:
                allowed_points_indices = []
                for part_i in range(self.num_parts):
                    if part_i not in visited_parts:
                        allowed_points_indices.extend([1 + 2 * part_i, 1 + 2 * part_i + 1])
                pheromones = self.pheromone[current_point_idx, allowed_points_indices] ** self.alpha
                visibilities = (
                    self.visibility[current_point_idx, allowed_points_indices] ** self.beta
                )
                probabilities = pheromones * visibilities
                prob_sum = np.sum(probabilities)
                if prob_sum == 0 or not np.isfinite(prob_sum):
                    next_point_idx = random.choice(allowed_points_indices)
                else:
                    probabilities /= prob_sum
                    next_point_idx = np.random.choice(allowed_points_indices, p=probabilities)
                chosen_part = (next_point_idx - 1) // 2
                chosen_dir = (next_point_idx - 1) % 2
                sequence.append(chosen_part)
                direction.append(chosen_dir)
                visited_parts.add(chosen_part)
                current_point_idx = next_point_idx + 1 if chosen_dir == 0 else next_point_idx - 1
            all_solutions.append(np.concatenate([sequence, direction]))
        return all_solutions

    def _update_pheromones(self, solutions, costs):
        """Preserve the source deposits between selected entry-node IDs."""
        self.pheromone *= 1 - self.rho
        for solution, cost in zip(solutions, costs):
            if cost == float("inf"):
                continue
            solution_array = np.array(solution)
            path_indices_in_coords = 2 * solution_array[: self.num_parts].astype(
                int
            ) + solution_array[self.num_parts :].astype(int)
            path_indices_in_all = [0] + (path_indices_in_coords + 1).tolist() + [0]
            for i in range(len(path_indices_in_all) - 1):
                p1_idx, p2_idx = (path_indices_in_all[i], path_indices_in_all[i + 1])
                delta_tau = self.q_val / cost
                self.pheromone[p1_idx, p2_idx] += delta_tau
                self.pheromone[p2_idx, p1_idx] += delta_tau

    def run_aco_for_instance(self, instance_data_tuple):
        start_time = time.time()
        if not self._initialize_instance_specifics(instance_data_tuple):
            return (None, float("inf"), float("inf"), 0.0)
        initial_nn_solution = self._create_individual_nn()
        self.current_initial_solution = initial_nn_solution
        self.current_initial_fitness = self._evaluate_solution(self.current_initial_solution)
        self.best_solution_overall = self.current_initial_solution
        self.best_fitness_overall = self.current_initial_fitness
        initial_cost = self.current_initial_fitness
        if self.best_fitness_overall != float("inf"):
            sol = self.best_solution_overall
            path_indices = 2 * sol[: self.num_parts].astype(int) + sol[self.num_parts :].astype(int)
            full_path = [0] + (path_indices + 1).tolist() + [0]
            delta_tau = self.q_val / self.best_fitness_overall
            for i in range(len(full_path) - 1):
                p1, p2 = (full_path[i], full_path[i + 1])
                self.pheromone[p1, p2] += delta_tau
                self.pheromone[p2, p1] += delta_tau
        for _ in range(self.generations):
            solutions = self._construct_ant_solutions()
            costs = np.array([self._evaluate_solution(sol) for sol in solutions])
            best_idx_in_gen = np.argmin(costs)
            if costs[best_idx_in_gen] < self.best_fitness_overall:
                self.best_fitness_overall = costs[best_idx_in_gen]
                self.best_solution_overall = solutions[best_idx_in_gen]
            self._update_pheromones(solutions, costs)
        duration = time.time() - start_time
        return (self.best_solution_overall, self.best_fitness_overall, initial_cost, duration)

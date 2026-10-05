"""Source-preserved optimizer; public execution is provided by src.run."""

import random
import time
import numpy as np


class GreyWolfOptimizerNesting:
    """Discrete GWO with optional adaptive, chaotic and state-aware updates."""

    def __init__(
        self,
        population_size=50,
        generations=100,
        initialize_method="random",
        mutation_rate_seq=0.02,
        mutation_rate_dir=0.01,
        apply_mutation_in_gwo_update=True,
        adaptive_a=False,
        adaptive_a_threshold=0.05,
        chaos_for_position_update=None,
        advanced_gwo=False,
    ):
        self.population_size = population_size
        self.generations = generations
        self.initialize_method = initialize_method
        self.mutation_rate_seq = mutation_rate_seq
        self.mutation_rate_dir = mutation_rate_dir
        self.apply_mutation_in_gwo_update = apply_mutation_in_gwo_update
        self.adaptive_a = adaptive_a
        self.adaptive_a_threshold = adaptive_a_threshold
        self.chaos_for_position_update = chaos_for_position_update
        self.advanced_gwo = advanced_gwo
        self.k_coupling = 0.05
        self.origin = None
        self.coords = None
        self.coords_paired = None
        self.num_parts = 0
        self.population = []
        self.fitness = np.zeros(population_size)
        self.best_solution_overall = None
        self.best_fitness_overall = float("inf")
        self.current_initial_solution = None
        self.current_initial_fitness = float("inf")
        self.chaos_z_for_pos = None

    def _process_input_data(self, data_tuple_raw):
        if data_tuple_raw is None or len(data_tuple_raw) < 3:
            self.origin, self.coords, self.coords_paired = (None, np.array([]), np.array([]))
            self.num_parts = 0
            return
        origin_np = np.array(data_tuple_raw[0], dtype=np.float32)
        coords_raw_np = np.array(data_tuple_raw[1], dtype=np.float32)
        coords_paired_raw_np = np.array(data_tuple_raw[2], dtype=np.float32)
        if coords_raw_np.ndim != 2 or coords_raw_np.shape[1] != 2:
            self.origin, self.coords, self.coords_paired = (
                origin_np,
                coords_raw_np,
                coords_paired_raw_np,
            )
            self.num_parts = 0
            return
        if coords_raw_np.size > 0:
            length_max = np.max(coords_raw_np)
            if length_max == 0:
                length_max = 1.0
            coords_normalized = coords_raw_np / length_max
            coords_paired_normalized = coords_paired_raw_np / length_max
        else:
            coords_normalized = coords_raw_np
            coords_paired_normalized = coords_paired_raw_np
        self.coords, self.coords_paired, self.origin = (
            coords_normalized,
            coords_paired_normalized,
            origin_np,
        )
        self.num_parts = int(self.coords.shape[0] // 2)

    def _initialize_instance_specifics(self, problem_instance_data_tuple):
        self._process_input_data(problem_instance_data_tuple)
        self.fitness = np.zeros(self.population_size)
        self.population = []
        self.best_solution_overall, self.best_fitness_overall = (None, float("inf"))
        self.current_initial_solution, self.current_initial_fitness = (None, float("inf"))
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

    def _create_individual_random(self):
        if self.num_parts == 0:
            return np.array([])
        sequence = np.random.permutation(self.num_parts)
        direction = np.random.choice([0, 1], size=self.num_parts)
        return np.concatenate([sequence, direction])

    def _create_individual_nn(self):
        if self.num_parts == 0 or self.coords.size == 0:
            return self._create_individual_random()
        all_points = np.concatenate([self.origin[np.newaxis, :], self.coords], axis=0)
        dist_matrix = np.linalg.norm(
            all_points[:, np.newaxis, :] - all_points[np.newaxis, :, :], axis=-1
        )
        visited_parts, sequence_nn, direction_nn = ([False] * self.num_parts, [], [])
        current_point_idx = 0
        for _ in range(self.num_parts):
            distances = dist_matrix[current_point_idx].copy()
            distances[[0, current_point_idx]] = np.inf
            for i, visited in enumerate(visited_parts):
                if visited:
                    distances[[2 * i + 1, 2 * i + 2]] = np.inf
            next_point_idx = np.argmin(distances)
            part_idx, is_A_side = ((next_point_idx - 1) // 2, (next_point_idx - 1) % 2 == 0)
            sequence_nn.append(part_idx)
            direction_nn.append(0 if is_A_side else 1)
            visited_parts[part_idx] = True
            current_point_idx = next_point_idx + (1 if is_A_side else -1)
        return np.concatenate([sequence_nn, direction_nn])

    def _initialize_population(self):
        self.population = []
        creator = (
            self._create_individual_nn
            if self.initialize_method == "nearest neighbor"
            else self._create_individual_random
        )
        first_individual = creator()
        if first_individual.size > 0:
            self.population.append(first_individual)
            self.current_initial_solution = np.copy(first_individual)
            self.current_initial_fitness = self._evaluate_solution(self.current_initial_solution)
        for _ in range(len(self.population), self.population_size):
            individual_to_add = creator()
            self.population.append(individual_to_add)

    def _mutate(self, individual_sol):
        if individual_sol is None or individual_sol.size == 0:
            return np.array([])
        mut_sol = np.copy(individual_sol)
        if random.random() < self.mutation_rate_seq and self.num_parts >= 2:
            m_idx1, m_idx2 = random.sample(range(self.num_parts), 2)
            mut_sol[m_idx1], mut_sol[m_idx2] = (mut_sol[m_idx2], mut_sol[m_idx1])
        for i in range(self.num_parts):
            if random.random() < self.mutation_rate_dir:
                mut_sol[self.num_parts + i] = 1 - mut_sol[self.num_parts + i]
        return mut_sol

    def _crossover(self, parent1_sol, parent2_sol):
        """Keep a leader segment and fill remaining sequence positions without duplicates."""
        size = self.num_parts
        if size == 0:
            return np.array([])
        p1_seq, p2_seq = (parent1_sol[:size].astype(int), parent2_sol[:size].astype(int))
        c_seq = np.full(size, -1, dtype=int)
        start, end = sorted(random.sample(range(size), 2))
        c_seq[start:end] = p1_seq[start:end]
        p2_idx = 0
        for i in range(size):
            if c_seq[i] == -1:
                while p2_seq[p2_idx] in c_seq:
                    p2_idx += 1
                c_seq[i] = p2_seq[p2_idx]
        return c_seq

    def _perturb_leader(self, leader_sol):
        C = 0.0
        if self.chaos_for_position_update == "logistic":
            if self.chaos_z_for_pos is None:
                self.chaos_z_for_pos = random.uniform(0.1, 0.9)
                while self.chaos_z_for_pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    self.chaos_z_for_pos = random.uniform(0.1, 0.9)
            self.chaos_z_for_pos = 4.0 * self.chaos_z_for_pos * (1 - self.chaos_z_for_pos)
            C = 2 * self.chaos_z_for_pos
        else:
            C = 2 * random.random()
        if C > 1.0 and self.num_parts >= 2:
            perturbed_sol = np.copy(leader_sol)
            idx1, idx2 = random.sample(range(self.num_parts), 2)
            perturbed_sol[idx1], perturbed_sol[idx2] = (perturbed_sol[idx2], perturbed_sol[idx1])
            return perturbed_sol
        else:
            return leader_sol

    def _create_individual_insertion(self, current_wolf):
        """Relocate a contiguous sequence block; retain direction bits by visit position."""
        size = self.num_parts
        if size < 2:
            return np.copy(current_wolf)
        mut_sol = np.copy(current_wolf)
        seq_part = mut_sol[:size].astype(int)
        idx1, idx2 = sorted(random.sample(range(size), 2))
        block = seq_part[idx1 : idx2 + 1]
        remainder = np.concatenate([seq_part[:idx1], seq_part[idx2 + 1 :]])
        insert_pos = random.randint(0, len(remainder))
        new_seq = np.concatenate([remainder[:insert_pos], block, remainder[insert_pos:]])
        mut_sol[:size] = new_seq
        return mut_sol

    def _create_individual_inversion(self, current_wolf):
        """Reverse a sequence segment; leave direction bits unchanged."""
        size = self.num_parts
        if size < 2:
            return np.copy(current_wolf)
        mut_sol = np.copy(current_wolf)
        seq_part = mut_sol[:size].astype(int)
        idx1, idx2 = sorted(random.sample(range(size), 2))
        seq_part[idx1 : idx2 + 1] = seq_part[idx1 : idx2 + 1][::-1]
        mut_sol[:size] = seq_part
        return mut_sol

    def _update_wolf_position(self, wolf, alpha, beta, delta, a_param, tau):
        """Generate one candidate and retain it only if travel strictly improves."""
        current_wolf = np.copy(wolf)
        current_fitness = self._evaluate_solution(current_wolf)
        new_wolf = None
        if self.advanced_gwo and random.random() < tau:
            r1 = random.random()
            A_val = 2.0 * a_param * r1 - a_param
            if abs(A_val) > 1.0:
                new_wolf = self._create_individual_insertion(current_wolf)
            else:
                new_wolf = self._create_individual_inversion(current_wolf)
        else:
            A = 2.0 * a_param * random.random() - a_param
            if abs(A) > 1:
                new_wolf = self._create_individual_insertion(current_wolf)
            elif self.advanced_gwo:
                leaders = [alpha, beta, delta]
                weights = [0.5, 0.3, 0.2]
                chosen_leader = random.choices(leaders, weights=weights, k=1)[0]
                perturbed_leader = self._perturb_leader(chosen_leader)
                new_seq = self._crossover(current_wolf, perturbed_leader)
                alpha_dir, beta_dir, delta_dir = (
                    alpha[self.num_parts :],
                    beta[self.num_parts :],
                    delta[self.num_parts :],
                )
                new_dir = np.zeros_like(alpha_dir)
                for i in range(len(new_dir)):
                    choice = random.randint(0, 2)
                    if choice == 0:
                        new_dir[i] = alpha_dir[i]
                    elif choice == 1:
                        new_dir[i] = beta_dir[i]
                    else:
                        new_dir[i] = delta_dir[i]
                new_wolf = np.concatenate([new_seq, new_dir])
            else:
                leaders = [alpha, beta, delta]
                weights = [0.5, 0.3, 0.2]
                chosen_leader = random.choices(leaders, weights=weights, k=1)[0]
                if self.chaos_for_position_update is not None:
                    perturbed_leader = self._perturb_leader(chosen_leader)
                else:
                    perturbed_leader = chosen_leader
                new_seq = self._crossover(current_wolf, perturbed_leader)
                alpha_dir, beta_dir, delta_dir = (
                    alpha[self.num_parts :],
                    beta[self.num_parts :],
                    delta[self.num_parts :],
                )
                new_dir = np.array(
                    [
                        random.choice([d1, d2, d3])
                        for d1, d2, d3 in zip(alpha_dir, beta_dir, delta_dir)
                    ]
                )
                new_wolf = np.concatenate([new_seq, new_dir])
        if self.apply_mutation_in_gwo_update:
            new_wolf = self._mutate(new_wolf)
        new_fitness = self._evaluate_solution(new_wolf)
        if new_fitness < current_fitness:
            return new_wolf
        else:
            return current_wolf

    def run_gwo_for_instance(self, instance_data_tuple):
        """Return the best route observed before each generation update."""
        if not self._initialize_instance_specifics(instance_data_tuple):
            return (None, float("inf"), float("inf"), 0.0)
        start_time = time.time()
        self._initialize_population()
        initial_cost = self.current_initial_fitness
        if not self.population:
            return (None, float("inf"), initial_cost, time.time() - start_time)
        if self.chaos_for_position_update is not None:
            self.chaos_z_for_pos = random.uniform(0.1, 0.9)
        for t in range(self.generations):
            self.fitness = np.array([self._evaluate_solution(wolf) for wolf in self.population])
            sorted_indices = np.argsort(self.fitness)
            alpha_idx, beta_idx, delta_idx = sorted_indices[0:3]
            alpha_wolf, beta_wolf, delta_wolf = (
                self.population[alpha_idx],
                self.population[beta_idx],
                self.population[delta_idx],
            )
            if self.fitness[alpha_idx] < self.best_fitness_overall:
                self.best_fitness_overall = self.fitness[alpha_idx]
                self.best_solution_overall = np.copy(alpha_wolf)
            a = 0.0
            tau = 0.0
            base_a = 2 - t * (2 / self.generations)
            if self.adaptive_a:
                mean_fitness = np.mean(self.fitness)
                if mean_fitness > 1e-09:
                    std_fitness = np.std(self.fitness)
                    normalized_diversity = std_fitness / mean_fitness
                    if normalized_diversity < self.adaptive_a_threshold:
                        if self.advanced_gwo:
                            stagnation_level = normalized_diversity / self.adaptive_a_threshold
                            a_boost = (1.0 - stagnation_level) * 1.5
                            a = min(base_a + a_boost, 2.0)
                        else:
                            a = 1.8
                    else:
                        a = base_a
                else:
                    a = base_a
            else:
                a = base_a
            if self.advanced_gwo:
                tau = self.k_coupling * a
            new_population = [np.copy(alpha_wolf), np.copy(beta_wolf), np.copy(delta_wolf)]
            other_wolves_indices = [
                i for i in range(len(self.population)) if i not in (alpha_idx, beta_idx, delta_idx)
            ]
            for i in other_wolves_indices:
                updated_wolf = self._update_wolf_position(
                    self.population[i], alpha_wolf, beta_wolf, delta_wolf, a_param=a, tau=tau
                )
                new_population.append(updated_wolf)
            self.population = new_population[: self.population_size]
            while len(self.population) < self.population_size:
                self.population.append(self._create_individual_random())
        duration = time.time() - start_time
        return (self.best_solution_overall, self.best_fitness_overall, initial_cost, duration)

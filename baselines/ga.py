"""Source-preserved optimizer; public execution is provided by src.run."""

import random
import time
import numpy as np


class GeneticAlgorithmNestingSAStyle:
    """Permutation/direction GA with tournament selection and elitism."""

    def __init__(
        self,
        population_size=50,
        generations=100,
        mutation_rate_seq=0.02,
        mutation_rate_dir=0.01,
        crossover_rate=0.9,
        elite_size=5,
        tournament_size=5,
        initialize_method="random",
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate_seq = mutation_rate_seq
        self.mutation_rate_dir = mutation_rate_dir
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.initialize_method = initialize_method
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
        self.coords = coords_normalized
        self.coords_paired = coords_paired_normalized
        self.origin = origin_np
        self.num_parts = int(self.coords.shape[0] // 2)
        if self.coords.shape[0] % 2 != 0 and self.coords.size > 0:
            pass

    def _initialize_instance_specifics(self, problem_instance_data_tuple):
        self._process_input_data(problem_instance_data_tuple)
        if self.num_parts == 0 and (self.coords is not None and self.coords.size > 0):
            pass
        self.fitness = np.zeros(self.population_size)
        self.population = []
        self.best_solution_overall = None
        self.best_fitness_overall = float("inf")
        self.current_initial_solution = None
        self.current_initial_fitness = float("inf")
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
            actual_part_indices_in_sequence = seq_part
            actual_directions_for_sequence = dir_part
            idx = 2 * actual_part_indices_in_sequence + actual_directions_for_sequence
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
        direction = np.random.choice(2, size=self.num_parts)
        return np.concatenate([sequence, direction])

    def _create_individual_nn(self):
        if (
            self.num_parts == 0
            or self.coords is None
            or self.coords.size == 0
            or (self.origin is None)
        ):
            return self._create_individual_random()
        coords_with_origin = np.concatenate([self.origin[np.newaxis, ...], self.coords], axis=0)
        if coords_with_origin.shape[0] <= 1:
            return self._create_individual_random()
        distance_matrix = np.linalg.norm(
            coords_with_origin[:, np.newaxis, :] - coords_with_origin[np.newaxis, :, :],
            ord=2,
            axis=-1,
        )
        np.fill_diagonal(distance_matrix, np.inf)
        distance_matrix[:, 0] = np.inf
        current_node_in_dist_matrix = 0
        sequence_nn = []
        direction_nn = []
        visited_unique_parts = [False] * self.num_parts
        for _ in range(self.num_parts):
            if np.all(np.isinf(distance_matrix[current_node_in_dist_matrix])):
                break
            next_node_overall_idx = np.argmin(distance_matrix[current_node_in_dist_matrix])
            selected_coord_row_idx = next_node_overall_idx - 1
            order = selected_coord_row_idx // 2
            dir_val = selected_coord_row_idx % 2
            temp_dist_row = np.copy(distance_matrix[current_node_in_dist_matrix])
            while order < 0 or order >= self.num_parts or visited_unique_parts[order]:
                temp_dist_row[next_node_overall_idx] = np.inf
                if np.all(np.isinf(temp_dist_row)):
                    order = -1
                    break
                next_node_overall_idx = np.argmin(temp_dist_row)
                selected_coord_row_idx = next_node_overall_idx - 1
                order = selected_coord_row_idx // 2
                dir_val = selected_coord_row_idx % 2
            if order == -1 or order < 0 or order >= self.num_parts:
                break
            sequence_nn.append(order)
            direction_nn.append(dir_val)
            visited_unique_parts[order] = True
            distance_matrix[:, selected_coord_row_idx + 1] = np.inf
            paired_coord_row_idx = selected_coord_row_idx ^ 1
            distance_matrix[:, paired_coord_row_idx + 1] = np.inf
            current_node_in_dist_matrix = paired_coord_row_idx + 1
            if current_node_in_dist_matrix >= distance_matrix.shape[0]:
                break
        if len(sequence_nn) < self.num_parts:
            remaining_parts_indices = [
                p for p in range(self.num_parts) if not visited_unique_parts[p]
            ]
            random.shuffle(remaining_parts_indices)
            sequence_nn.extend(remaining_parts_indices)
            num_dirs_needed = self.num_parts - len(direction_nn)
            if num_dirs_needed > 0:
                direction_nn.extend(np.random.choice(2, size=num_dirs_needed).tolist())
        if not sequence_nn:
            return self._create_individual_random()
        final_sequence = np.array(sequence_nn[: self.num_parts])
        final_direction = np.array(direction_nn[: self.num_parts])
        return np.concatenate([final_sequence, final_direction])

    def _initialize_population(self):
        self.population = []
        base_individual_for_population_creation = None
        if self.initialize_method == "nearest neighbor":
            first_individual = self._create_individual_nn()
            if first_individual.size > 0:
                base_individual_for_population_creation = first_individual
        elif self.initialize_method == "random":
            first_individual = self._create_individual_random()
        else:
            first_individual = self._create_individual_random()
        if first_individual is not None and first_individual.size > 0:
            self.population.append(first_individual)
            self.current_initial_solution = np.copy(first_individual)
            self.current_initial_fitness = self._evaluate_solution(self.current_initial_solution)
        else:
            self.current_initial_solution = None
            self.current_initial_fitness = float("inf")
            base_individual_for_population_creation = None
        current_pop_count = len(self.population)
        for _ in range(current_pop_count, self.population_size):
            individual_to_add = None
            if (
                self.initialize_method == "nearest neighbor"
                and base_individual_for_population_creation is not None
                and (base_individual_for_population_creation.size > 0)
            ):
                mutated_individual = self._mutate(np.copy(base_individual_for_population_creation))
                if mutated_individual.size > 0:
                    individual_to_add = mutated_individual
                else:
                    individual_to_add = self._create_individual_random()
            else:
                individual_to_add = self._create_individual_random()
            if individual_to_add is not None and individual_to_add.size > 0:
                self.population.append(individual_to_add)
        if not self.population and self.population_size > 0:
            pass

    def _evaluate_population(self):
        if not self.population:
            self.best_fitness_overall = float("inf")
            return
        for i, individual in enumerate(self.population):
            self.fitness[i] = self._evaluate_solution(individual)
        if self.fitness.size > 0 and len(self.population) > 0:
            valid_fitness_indices = np.where(self.fitness != float("inf"))[0]
            if valid_fitness_indices.size > 0:
                current_best_idx_pop = valid_fitness_indices[
                    np.argmin(self.fitness[valid_fitness_indices])
                ]
                if self.fitness[current_best_idx_pop] < self.best_fitness_overall:
                    self.best_fitness_overall = self.fitness[current_best_idx_pop]
                    self.best_solution_overall = np.copy(self.population[current_best_idx_pop])

    def _tournament_selection(self):
        mating_pool = []
        if not self.population or len(self.population) < self.tournament_size:
            return [np.copy(ind) for ind in self.population]
        for _ in range(self.population_size):
            try:
                participants_indices = random.sample(
                    range(len(self.population)), self.tournament_size
                )
                winner_idx = -1
                min_fitness_val = float("inf")
                for idx_sel in participants_indices:
                    if self.fitness[idx_sel] < min_fitness_val:
                        min_fitness_val = self.fitness[idx_sel]
                        winner_idx = idx_sel
                if winner_idx != -1:
                    mating_pool.append(self.population[winner_idx])
                elif self.population:
                    mating_pool.append(self.population[random.choice(participants_indices)])
            except ValueError:
                if self.population:
                    mating_pool.append(random.choice(self.population))
                else:
                    break
        return mating_pool

    def _crossover(self, parent1_sol, parent2_sol):
        """Position-based permutation crossover with uniform direction exchange."""
        p1_seq, p1_dir = (
            parent1_sol[: self.num_parts].astype(int),
            parent1_sol[self.num_parts :].astype(int),
        )
        p2_seq, p2_dir = (
            parent2_sol[: self.num_parts].astype(int),
            parent2_sol[self.num_parts :].astype(int),
        )
        size_seq = self.num_parts
        c1_s = np.full(size_seq, -1, dtype=int)
        if size_seq > 0:
            num_pos_to_inherit_c1 = random.randint(1, size_seq)
            positions_from_p1 = sorted(random.sample(range(size_seq), k=num_pos_to_inherit_c1))
            for pos in positions_from_p1:
                c1_s[pos] = p1_seq[pos]
            p2_fill_idx = 0
            for i in range(size_seq):
                if c1_s[i] == -1:
                    while p2_seq[p2_fill_idx] in c1_s:
                        p2_fill_idx += 1
                    c1_s[i] = p2_seq[p2_fill_idx]
                    p2_fill_idx += 1
        else:
            c1_s = np.array([], dtype=int)
        c2_s = np.full(size_seq, -1, dtype=int)
        if size_seq > 0:
            num_pos_to_inherit_c2 = random.randint(1, size_seq)
            positions_from_p2 = sorted(random.sample(range(size_seq), k=num_pos_to_inherit_c2))
            for pos in positions_from_p2:
                c2_s[pos] = p2_seq[pos]
            p1_fill_idx = 0
            for i in range(size_seq):
                if c2_s[i] == -1:
                    while p1_seq[p1_fill_idx] in c2_s:
                        p1_fill_idx += 1
                    c2_s[i] = p1_seq[p1_fill_idx]
                    p1_fill_idx += 1
        else:
            c2_s = np.array([], dtype=int)
        c1_d, c2_d = (np.copy(p1_dir), np.copy(p2_dir))
        if self.num_parts > 0:
            for i in range(self.num_parts):
                if random.random() < 0.5:
                    c1_d[i], c2_d[i] = (c2_d[i], c1_d[i])
        else:
            c1_d = np.array([], dtype=int)
            c2_d = np.array([], dtype=int)
        child1 = np.concatenate([c1_s, c1_d])
        child2 = np.concatenate([c2_s, c2_d])
        return (child1, child2)

    def _mutate(self, individual_sol):
        if individual_sol is None or individual_sol.size == 0:
            return np.array([])
        mut_sol = np.copy(individual_sol)
        if random.random() < self.mutation_rate_seq:
            if self.num_parts >= 2:
                m_idx1, m_idx2 = random.sample(range(self.num_parts), 2)
                mut_sol[m_idx1], mut_sol[m_idx2] = (mut_sol[m_idx2], mut_sol[m_idx1])
        for i in range(self.num_parts):
            if random.random() < self.mutation_rate_dir:
                mut_sol[self.num_parts + i] = 1 - mut_sol[self.num_parts + i]
        return mut_sol

    def run_ga_for_instance(self, instance_data_tuple):
        if not self._initialize_instance_specifics(instance_data_tuple):
            return (None, float("inf"), float("inf"), 0.0)
        start_run_time = time.time()
        self._initialize_population()
        initial_objective_value_for_run = self.current_initial_fitness
        if not self.population or (self.population[0].size == 0 and self.num_parts > 0):
            return (
                None,
                self.best_fitness_overall,
                initial_objective_value_for_run,
                time.time() - start_run_time,
            )
        self._evaluate_population()
        for gen_idx_run_loop in range(self.generations):
            mating_pool_run_loop = self._tournament_selection()
            if not mating_pool_run_loop:
                break
            new_pop_run_loop = []
            if self.fitness.size > 0 and self.elite_size > 0 and (len(self.population) > 0):
                valid_fitness_indices = np.where(self.fitness != float("inf"))[0]
                if valid_fitness_indices.size > 0:
                    sorted_valid_indices = valid_fitness_indices[
                        np.argsort(self.fitness[valid_fitness_indices])
                    ]
                    elite_indices_run_loop = sorted_valid_indices[
                        : min(self.elite_size, len(sorted_valid_indices))
                    ]
                    for idx_e_run_loop in elite_indices_run_loop:
                        new_pop_run_loop.append(np.copy(self.population[idx_e_run_loop]))
            needed_offspring_run_loop = self.population_size - len(new_pop_run_loop)
            offspring_list_run_loop = []
            if len(mating_pool_run_loop) >= 2 and needed_offspring_run_loop > 0:
                pool_idx_run_loop = 0
                while len(offspring_list_run_loop) < needed_offspring_run_loop:
                    p1_ga = mating_pool_run_loop[pool_idx_run_loop % len(mating_pool_run_loop)]
                    p2_ga = mating_pool_run_loop[
                        (pool_idx_run_loop + 1) % len(mating_pool_run_loop)
                    ]
                    pool_idx_run_loop = pool_idx_run_loop + 2
                    if random.random() < self.crossover_rate:
                        c1_ga, c2_ga = self._crossover(p1_ga, p2_ga)
                    else:
                        c1_ga, c2_ga = (np.copy(p1_ga), np.copy(p2_ga))
                    offspring_list_run_loop.append(self._mutate(c1_ga))
                    if len(offspring_list_run_loop) < needed_offspring_run_loop:
                        offspring_list_run_loop.append(self._mutate(c2_ga))
            elif mating_pool_run_loop and needed_offspring_run_loop > 0:
                while (
                    len(offspring_list_run_loop) < needed_offspring_run_loop
                    and mating_pool_run_loop
                ):
                    offspring_list_run_loop.append(
                        self._mutate(np.copy(random.choice(mating_pool_run_loop)))
                    )
            new_pop_run_loop.extend(offspring_list_run_loop)
            if not new_pop_run_loop and self.population_size > 0:
                break
            self.population = [ind for ind in new_pop_run_loop if ind.size > 0][
                : self.population_size
            ]
            if self.population:
                self._evaluate_population()
            else:
                break
        duration_run_final_val = time.time() - start_run_time
        return (
            self.best_solution_overall,
            self.best_fitness_overall,
            initial_objective_value_for_run,
            duration_run_final_val,
        )

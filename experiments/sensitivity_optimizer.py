"""Source-preserved optimizer; public execution is provided by src.run."""

import random
import time
import numpy as np


class GreyWolfOptimizerNesting:
    """Distinct uploaded sensitivity implementation; retained independently."""

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
        k_coupling=0.05,
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
        self.k_coupling = k_coupling
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
        seq_part = solution_array[: self.num_parts].astype(int)
        dir_part = solution_array[self.num_parts :].astype(int)
        try:
            idx = 2 * seq_part + dir_part
            coords_A_sides = self.coords[idx]
            coords_B_sides = self.coords_paired[idx]
        except:
            return float("inf")
        starts = np.concatenate([coords_A_sides, self.origin[np.newaxis, ...]], axis=0)
        ends = np.concatenate([self.origin[np.newaxis, ...], coords_B_sides], axis=0)
        return np.linalg.norm(ends - starts, ord=2, axis=-1).sum()

    def _create_individual_random(self):
        if self.num_parts == 0:
            return np.array([])
        return np.concatenate(
            [np.random.permutation(self.num_parts), np.random.choice([0, 1], size=self.num_parts)]
        )

    def _create_individual_nn(self):
        if self.num_parts == 0 or self.coords.size == 0:
            return self._create_individual_random()
        all_p = np.concatenate([self.origin[np.newaxis, :], self.coords], axis=0)
        dist_m = np.linalg.norm(all_p[:, np.newaxis, :] - all_p[np.newaxis, :, :], axis=-1)
        visited, seq, dr = ([False] * self.num_parts, [], [])
        curr = 0
        for _ in range(self.num_parts):
            dists = dist_m[curr].copy()
            dists[[0, curr]] = np.inf
            for i, v in enumerate(visited):
                if v:
                    dists[[2 * i + 1, 2 * i + 2]] = np.inf
            next_p = np.argmin(dists)
            p_idx, is_A = ((next_p - 1) // 2, (next_p - 1) % 2 == 0)
            seq.append(p_idx)
            dr.append(0 if is_A else 1)
            visited[p_idx] = True
            curr = next_p + (1 if is_A else -1)
        return np.concatenate([seq, dr])

    def _initialize_population(self):
        self.population = []
        creator = (
            self._create_individual_nn
            if self.initialize_method == "nearest neighbor"
            else self._create_individual_random
        )
        self.population.append(creator())
        self.current_initial_fitness = self._evaluate_solution(self.population[0])
        for _ in range(1, self.population_size):
            self.population.append(creator())

    def _mutate(self, individual_sol):
        mut_sol = np.copy(individual_sol)
        if random.random() < self.mutation_rate_seq and self.num_parts >= 2:
            i, j = random.sample(range(self.num_parts), 2)
            mut_sol[i], mut_sol[j] = (mut_sol[j], mut_sol[i])
        for i in range(self.num_parts):
            if random.random() < self.mutation_rate_dir:
                mut_sol[self.num_parts + i] = 1 - mut_sol[self.num_parts + i]
        return mut_sol

    def _crossover(self, p1, p2):
        size = self.num_parts
        p1_seq, p2_seq = (p1[:size].astype(int), p2[:size].astype(int))
        c_seq = np.full(size, -1, dtype=int)
        a, b = sorted(random.sample(range(size), 2))
        c_seq[a:b] = p1_seq[a:b]
        p2_ptr = 0
        for i in range(size):
            if c_seq[i] == -1:
                while p2_seq[p2_ptr] in c_seq:
                    p2_ptr += 1
                c_seq[i] = p2_seq[p2_ptr]
        return c_seq

    def _perturb_leader(self, leader):
        if self.chaos_for_position_update == "logistic":
            if self.chaos_z_for_pos is None:
                self.chaos_z_for_pos = random.uniform(0.1, 0.9)
            self.chaos_z_for_pos = 4.0 * self.chaos_z_for_pos * (1 - self.chaos_z_for_pos)
            C = 2 * self.chaos_z_for_pos
        else:
            C = 2 * random.random()
        if C > 1.0 and self.num_parts >= 2:
            sol = np.copy(leader)
            i, j = random.sample(range(self.num_parts), 2)
            sol[i], sol[j] = (sol[j], sol[i])
            return sol
        return leader

    def _update_wolf_position(self, wolf, alpha, beta, delta, a_param, tau):
        curr_wolf = np.copy(wolf)
        curr_fit = self._evaluate_solution(curr_wolf)
        if self.advanced_gwo and random.random() < tau:
            A_val = 2.0 * a_param * random.random() - a_param
            new_wolf = (
                self._create_individual_insertion(curr_wolf)
                if abs(A_val) > 1.0
                else self._create_individual_inversion(curr_wolf)
            )
        else:
            A = 2.0 * a_param * random.random() - a_param
            if abs(A) > 1:
                new_wolf = self._create_individual_insertion(curr_wolf)
            else:
                leaders = [alpha, beta, delta]
                weights = [0.5, 0.3, 0.2]
                chosen = random.choices(leaders, weights=weights, k=1)[0]
                new_seq = self._crossover(curr_wolf, self._perturb_leader(chosen))
                a_d, b_d, d_d = (
                    alpha[self.num_parts :],
                    beta[self.num_parts :],
                    delta[self.num_parts :],
                )
                new_dir = np.array(
                    [random.choice([d1, d2, d3]) for d1, d2, d3 in zip(a_d, b_d, d_d)]
                )
                new_wolf = np.concatenate([new_seq, new_dir])
        if self.apply_mutation_in_gwo_update:
            new_wolf = self._mutate(new_wolf)
        return new_wolf if self._evaluate_solution(new_wolf) < curr_fit else curr_wolf

    def _create_individual_insertion(self, wolf):
        size = self.num_parts
        if size < 2:
            return wolf
        sol = np.copy(wolf)
        seq = sol[:size]
        i, j = sorted(random.sample(range(size), 2))
        block = seq[i : j + 1]
        rem = np.concatenate([seq[:i], seq[j + 1 :]])
        ins = random.randint(0, len(rem))
        sol[:size] = np.concatenate([rem[:ins], block, rem[ins:]])
        return sol

    def _create_individual_inversion(self, wolf):
        size = self.num_parts
        if size < 2:
            return wolf
        sol = np.copy(wolf)
        seq = sol[:size]
        i, j = sorted(random.sample(range(size), 2))
        seq[i : j + 1] = seq[i : j + 1][::-1]
        sol[:size] = seq
        return sol

    def run_gwo_for_instance(self, instance_data):
        if not self._initialize_instance_specifics(instance_data):
            return (None, float("inf"), float("inf"), 0.0)
        start_t = time.time()
        self._initialize_population()
        init_cost = self.current_initial_fitness
        for t in range(self.generations):
            self.fitness = np.array([self._evaluate_solution(w) for w in self.population])
            idx = np.argsort(self.fitness)
            a_w, b_w, d_w = (
                self.population[idx[0]],
                self.population[idx[1]],
                self.population[idx[2]],
            )
            if self.fitness[idx[0]] < self.best_fitness_overall:
                self.best_fitness_overall = self.fitness[idx[0]]
                self.best_solution_overall = np.copy(a_w)
            base_a = 2 - t * (2 / self.generations)
            a = base_a
            if self.adaptive_a:
                mu = np.mean(self.fitness)
                if mu > 1e-09:
                    div = np.std(self.fitness) / mu
                    if div < self.adaptive_a_threshold:
                        if self.advanced_gwo:
                            a = min(base_a + (1.0 - div / self.adaptive_a_threshold) * 1.5, 2.0)
                        else:
                            a = 1.8
            tau = self.k_coupling * a if self.advanced_gwo else 0.0
            new_pop = [np.copy(a_w), np.copy(b_w), np.copy(d_w)]
            for i in range(3, self.population_size):
                new_pop.append(
                    self._update_wolf_position(self.population[i], a_w, b_w, d_w, a, tau)
                )
            self.population = new_pop
        return (
            self.best_solution_overall,
            self.best_fitness_overall,
            init_cost,
            time.time() - start_t,
        )

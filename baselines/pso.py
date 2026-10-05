"""Source-preserved optimizer; public execution is provided by src.run."""

import random
import time
import numpy as np
import scipy.special


class PSOBase:
    """Random-key representation and common route evaluation."""

    def __init__(self, population_size, generations):
        self.pop_size = population_size
        self.generations = generations
        self.num_parts = 0
        self.dim = 0
        self.origin = None
        self.coords = None
        self.coords_paired = None
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_fitness = None
        self.gbest_position = None
        self.gbest_fitness = float("inf")

    def _initialize_instance_specifics(self, problem_instance_data_tuple):
        if not problem_instance_data_tuple or len(problem_instance_data_tuple) < 3:
            return False
        origin_data, coords_data, coords_paired_data = problem_instance_data_tuple
        self.origin = np.array(origin_data, dtype=np.float32)
        self.coords = np.array(coords_data, dtype=np.float32)
        self.coords_paired = np.array(coords_paired_data, dtype=np.float32)
        if self.coords.size == 0 or self.coords.ndim != 2:
            return False
        self.num_parts = self.coords.shape[0] // 2
        if self.num_parts == 0:
            return False
        self.dim = self.num_parts * 2
        length_max = np.max(self.coords) if self.coords.size > 0 else 1.0
        if length_max == 0:
            length_max = 1.0
        self.coords /= length_max
        self.coords_paired /= length_max
        self.origin /= length_max
        self.positions = np.random.rand(self.pop_size, self.dim)
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.pop_size, self.dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(self.pop_size, float("inf"))
        self.gbest_fitness = float("inf")
        self.gbest_position = None
        return True

    def _evaluate_fitness(self, position):
        if self.num_parts == 0:
            return float("inf")
        seq_part = np.argsort(position[: self.num_parts])
        dir_part = (position[self.num_parts :] > 0.5).astype(np.int64)
        try:
            idx = 2 * seq_part + dir_part
            coords_A_sides = self.coords[idx]
            coords_B_sides = self.coords_paired[idx]
            starts_points = np.vstack([self.origin, coords_B_sides[:-1]])
            end_points = coords_A_sides
            cost = np.linalg.norm(end_points - starts_points, axis=1).sum()
            cost += np.linalg.norm(coords_B_sides[-1] - self.origin)
            return cost
        except (IndexError, TypeError):
            return float("inf")

    def _create_individual_nn(self):
        if (
            self.num_parts == 0
            or self.coords is None
            or self.coords.size == 0
            or (self.origin is None)
        ):
            return np.array([])
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
            for i in range(self.num_parts):
                if visited_parts[i]:
                    distances[2 * i + 1] = np.inf
                    distances[2 * i + 2] = np.inf
            next_point_idx = np.argmin(distances[1:]) + 1
            part_idx = (next_point_idx - 1) // 2
            sequence_nn.append(part_idx)
            direction_nn.append((next_point_idx - 1) % 2)
            visited_parts[part_idx] = True
            current_point_idx = part_idx * 2 + 1 + ((next_point_idx - 1) % 2 == 0)
        return np.concatenate([sequence_nn, direction_nn])

    def _mutate(self, individual_sol, mutation_rate_seq, mutation_rate_dir):
        if individual_sol is None or individual_sol.size == 0:
            return np.array([])
        mut_sol = np.copy(individual_sol)
        if self.num_parts >= 2 and random.random() < mutation_rate_seq:
            m_idx1, m_idx2 = random.sample(range(self.num_parts), 2)
            mut_sol[m_idx1], mut_sol[m_idx2] = (mut_sol[m_idx2], mut_sol[m_idx1])
        for i in range(self.num_parts):
            if random.random() < mutation_rate_dir:
                mut_sol[self.num_parts + i] = 1 - mut_sol[self.num_parts + i]
        return mut_sol

    def run(self, instance_data_tuple):
        raise NotImplementedError


class LPSPOOptimizer(PSOBase):
    """Uploaded LPSPSO variant with Singer-map and Levy-flight control."""

    def __init__(
        self,
        population_size,
        generations,
        use_nn_init=False,
        mutation_rate_seq=0.05,
        mutation_rate_dir=0.02,
    ):
        super().__init__(population_size, generations)
        self.use_nn_init = use_nn_init
        self.mutation_rate_seq = mutation_rate_seq
        self.mutation_rate_dir = mutation_rate_dir
        self.w_max, self.w_min = (0.95, 0.4)
        self.c1_a, self.c2_b = (1.25, 2.5)
        self.singer_mu = 1.04
        self.levy_beta = 1.5
        self.singer_x = random.uniform(0.1, 0.9)

    def _singer_map_update(self):
        x = self.singer_x
        self.singer_x = self.singer_mu * (7.86 * x - 23.3 * x**2 + 28.75 * x**3 - 13.3 * x**4)
        return np.clip(self.singer_x, 0, 1)

    def _get_levy_flight_step(self):
        beta = self.levy_beta
        num = scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma_u = (num / den) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def _initialize_population_nn(self, instance_data_tuple):
        if not self._initialize_instance_specifics(instance_data_tuple):
            return False
        nn_solution = self._create_individual_nn()
        if nn_solution is None or nn_solution.size == 0:
            return False
        nn_position = np.zeros(self.dim)
        nn_position[nn_solution[: self.num_parts]] = np.arange(self.num_parts) / self.num_parts
        nn_position[self.num_parts :] = nn_solution[self.num_parts :] * 0.5 + 0.25
        self.positions = np.zeros((self.pop_size, self.dim))
        self.positions[0] = nn_position
        for i in range(1, self.pop_size):
            mutated_sol = self._mutate(nn_solution, self.mutation_rate_seq, self.mutation_rate_dir)
            mut_position = np.zeros(self.dim)
            if mutated_sol.size > 0:
                mut_position[mutated_sol[: self.num_parts]] = (
                    np.arange(self.num_parts) / self.num_parts
                )
                mut_position[self.num_parts :] = mutated_sol[self.num_parts :] * 0.5 + 0.25
                self.positions[i] = mut_position
            else:
                self.positions[i] = np.random.rand(self.dim)
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.pop_size, self.dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(self.pop_size, float("inf"))
        self.gbest_fitness = float("inf")
        self.gbest_position = None
        return True

    def run(self, instance_data_tuple):
        start_time = time.time()
        if self.use_nn_init:
            if not self._initialize_population_nn(instance_data_tuple):
                return (float("inf"), 0.0, float("inf"))
        elif not self._initialize_instance_specifics(instance_data_tuple):
            return (float("inf"), 0.0, float("inf"))
        t_max = self.generations
        initial_gbest = float("inf")
        for i in range(self.pop_size):
            fitness = self._evaluate_fitness(self.positions[i])
            self.pbest_fitness[i] = fitness
            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()
        initial_gbest = self.gbest_fitness
        for t in range(t_max):
            w = (self.w_max + self.w_min) / 2 + (self.w_max - self.w_min) / 2 * np.exp(-t / t_max)
            c1 = self.c1_a + np.exp(-t / t_max)
            c2 = self.c2_b - c1
            if self.num_parts > 0:
                d_avgs = [
                    np.mean(
                        np.linalg.norm(
                            self.positions[i] - np.delete(self.positions, i, axis=0), axis=1
                        )
                    )
                    for i in range(self.pop_size)
                ]
                d_max, d_min, d_avg = (np.max(d_avgs), np.min(d_avgs), np.mean(d_avgs))
                kappa = (d_avg - d_min) / (d_max - d_min + 1e-10)
                tau = np.exp(-(t + 1) / t_max)
            else:
                kappa, tau = (0, 1)
            for i in range(self.pop_size):
                if kappa > tau and self.gbest_position is not None:
                    step = self._get_levy_flight_step()
                    self.positions[i] += 0.01 * step * (self.positions[i] - self.gbest_position)
                else:
                    r1 = np.random.rand(self.dim)
                    r2_chaotic = self._singer_map_update()
                    cognitive = c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                    social = (
                        c2 * r2_chaotic * (self.gbest_position - self.positions[i])
                        if self.gbest_position is not None
                        else 0
                    )
                    self.velocities[i] = w * self.velocities[i] + cognitive + social
                    self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], 0, 1)
                fitness = self._evaluate_fitness(self.positions[i])
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
        duration = time.time() - start_time
        return (self.gbest_fitness, duration, initial_gbest)

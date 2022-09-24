import sys
sys.path.insert(0, 'evoman')
# imports other libs
import time, random
import numpy as np
from math import fabs,sqrt
import os
from specialist_config import SpecialistConfig


class OptimizationSpecialist:
    def __init__(self):
        self.config = SpecialistConfig()
        self.env = self.config.init_environment()
        self.env.state_to_log()
        self.n_weights = (self.env.get_num_sensors() + 1) * self.config.n_hidden_neurons + (self.config.n_hidden_neurons + 1) * 5

    def load_experiment(self):
        # initializes population loading old solutions or generating new ones
        if not os.path.exists(self.config.experiment_name + '/evoman_solstate'):
            print('\nNEW EVOLUTION\n')
            pop = np.random.uniform(self.config.lower_limit,
                                    self.config.upp_limit,
                                    (self.config.n_pop, self.n_weights))
            fit_pop = self.evaluate(pop)
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            ini_g = 0
            solutions = [pop, fit_pop]
            self.env.update_solutions(solutions)
        else:
            print('\nCONTINUING EVOLUTION\n')
            self.env.load_state()
            pop = self.env.solutions[0]
            fit_pop = self.env.solutions[1]
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            # finds last generation number
            with open(self.config.experiment_name + '/gen.txt', 'r') as f:
                ini_g = int(f.readline())

        return pop, fit_pop, best, mean, std, ini_g

    def simulation(self, env, x):
        f, p, e, t = env.play(pcont=x)
        return f

    def normalize(self, x, pfit_pop):
        if (max(pfit_pop) - min(pfit_pop)) > 0:
            x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
        else:
            x_norm = 0
        if x_norm <= 0:
            x_norm = 0.0000000001
        return x_norm

    def evaluate(self, x):
        return np.array(list(map(lambda y: self.simulation(self.env, y), x)))

    def tournament2(self, pop, fit_pop, k=2):
        selected = []
        for i in range(k):
            children_indices = np.random.randint(0, pop.shape[0], self.config.tournament_size)
            children_fit = [fit_pop[i] for i in children_indices]
            fittest_child = np.argmax(children_fit)
            fittest_index = children_indices[fittest_child]
            selected.append(pop[fittest_index])
        return selected

    def tournament1(self, pop, fit_pop):
        c1 = np.random.randint(0, pop.shape[0], 1)
        c2 = np.random.randint(0, pop.shape[0], 1)

        if fit_pop[c1] > fit_pop[c2]:
            return pop[c1][0]
        else:
            return pop[c2][0]


    def limits(self, x):
        if x > self.config.upp_limit:
            return self.config.upp_limit
        elif x < self.config.lower_limit:
            return self.config.lower_limit
        else:
            return x

    def twoPointCrossover(self, p1, p2):
        size = min(len(p1), len(p2))
        point1 = np.random.randint(1, size)
        point2 = np.random.randint(1, size - 1)
        if point2 >= point1:
            point2 += 1
        else:
            point1, point2 = point2, point1
        p1[point1:point2], p2[point1:point2] = p2[point1:point2], p1[point1:point2]
        return p1, p2

    def kPointCrossover(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        k = self.config.k_points
        if k > size:
            k = size
        cutoff_points = []
        while (len(cutoff_points) < k):
            num = np.random.randint(0, size)
            if num not in cutoff_points:
                cutoff_points.append(num)
        cutoff_points.sort()
        for i in range(len(cutoff_points)):
            p1 = cutoff_points[i]
            ind1[p1::], ind2[p1::] = ind2[p1::], ind1[p1::]
        return ind1, ind2

    def mutGuass(self, offsp):
        size = len(offsp)
        for i in range(size):
            if np.random.random() < self.config.mutation_prob:
                offsp[i] += random.gauss(self.config.mu, self.config.sigma)
        return offsp

    def mutPolinomialBounded(self, offsp, eta=0.35, lower_bound=0, upper_bound=1):
        """Polinomial bounded mutation implemetation taken from the DEAP framework"""
        size = len(offsp)
        for i, xl, xu in zip(range(size), lower_bound, upper_bound):
            if random.random() <= self.config.mutation_prob:
                x = offsp[i]
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)
                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow
                x = x + delta_q * (xu - xl)
                x = min(max(x, xl), xu)
                offsp[i] = x
        return offsp

    # crossover
    def kissland(self, pop, fit_pop):
        total_offspring = np.zeros((0, self.n_weights))
        for p in range(0, pop.shape[0], 2):
            # Selection
            # parent1, parent2 = self.tournament(pop, fit_pop)
            parent1 = self.tournament1(pop, fit_pop)
            parent2 = self.tournament1(pop, fit_pop)
            n_offsp = np.random.randint(1, 4, 1)[0]
            offsp = np.zeros((n_offsp, self.n_weights))
            for f in range(0, n_offsp):
                # Crossover
                if self.config.crossover_algorithm == 'two_point':
                    parent1, parent2 = self.twoPointCrossover(parent1, parent2)
                elif self.config.crossover_algorithm == 'k_point':
                    parent1, parent2 = self.kPointCrossover(parent1, parent2)
                offsp[f] = parent1 + parent2
                # Mutation
                if self.config.mutation_algorithm == "gauss":
                    final_offsp  = self.mutGuass(offsp)
                elif self.config.mutation_algorithm == "polinomial":
                    final_offsp = self.mutPolinomialBounded(offsp)
        return final_offsp

    # kills the worst genomes, and replace with new best/random solutions
    def purge(self, pop, fit_pop):
        worst = int(self.config.n_pop / 4)  # a quarter of the population
        order = np.argsort(fit_pop)
        orderasc = order[0:worst]
        for o in orderasc:
            for j in range(0, self.n_weights):
                pro = np.random.uniform(0, 1)
                if np.random.uniform(0, 1) <= pro:
                    pop[o][j] = np.random.uniform(self.config.upp_limit, self.config.lower_limit)  # random dna, uniform dist.
                else:
                    pop[o][j] = pop[order[-1:]][0][j]  # dna from best
            fit_pop[o] = self.evaluate([pop[o]])
        return pop, fit_pop

    def run(self):
        start = time.time()
        pop, fit_pop, best, mean, std, ini_g = self.load_experiment()
        if self.config.run_mode == 'test':
            bsol = np.loadtxt(self.config.experiment_name + '/best.txt')
            print('\n RUNNING SAVED BEST SOLUTION \n')
            self.env.update_parameter('speed', 'normal')
            self.evaluate([bsol])
            sys.exit(0)
        # saves results for first pop
        with open(self.config.experiment_name + '/results.txt', 'a') as f:
            f.write('\n gen | best | mean | std')
            print(f'\n GENERATION {str(ini_g)} - {str(round(fit_pop[best], 6))} {str(round(mean, 6))} {str(round(std, 6))}')
            f.write(f'\n GENERATION {str(ini_g)} | {str(round(fit_pop[best], 6))} | {str(round(mean, 6))} | {str(round(std, 6))}')

        last_sol = fit_pop[best]
        not_improved = 0
        for i in range(ini_g + 1, self.config.generations):
            offspring = self.kissland(pop, fit_pop)
            fit_offspring = self.evaluate(offspring)
            pop = np.vstack((pop, offspring))
            fit_pop = np.append(fit_pop, fit_offspring)

            best = np.argmax(fit_pop)
            fit_pop[best] = float(self.evaluate(np.array([pop[best]]))[0])
            best_sol = fit_pop[best]

            # selection
            fit_pop_cp = fit_pop
            fit_pop_norm = np.array(list(map(lambda y: self.normalize(y, fit_pop_cp), fit_pop)))
            probs = (fit_pop_norm) / (fit_pop_norm).sum()
            chosen = np.random.choice(pop.shape[0], self.config.n_pop, p=probs, replace=False)
            chosen = np.append(chosen[1:], best)
            pop = pop[chosen]
            fit_pop = fit_pop[chosen]

            # searching new areas

            if best_sol <= last_sol:
                not_improved += 1
            else:
                last_sol = best_sol
            not_improved = 0

            if not_improved >= 15:
                with open(self.config.experiment_name + '/results.txt', 'a') as f:
                    f.write('\n purge')

            pop, fit_pop = self.purge(pop, fit_pop)
            not_improved = 0

            best = np.argmax(fit_pop)
            std = np.std(fit_pop)
            mean = np.mean(fit_pop)

            # saves results
            with open(self.config.experiment_name + '/results.txt', 'a') as f:
                print(f'\n GENERATION {str(i)} {str(round(fit_pop[best], 6))} {str(round(mean, 6))} {str(round(std, 6))}')
                f.write(f'\n GENERATION {str(i)} | {str(round(fit_pop[best], 6))} | {str(round(mean, 6))} | {str(round(std, 6))}')

            # saves generation number
            with open(self.config.experiment_name + '/gen.txt', 'w') as f:
                f.write(str(i))

            # saves file with the best solution
            np.savetxt(self.config.experiment_name + '/best.txt', pop[best])

            # saves simulation state
            solutions = [pop, fit_pop]
            self.env.update_solutions(solutions)
            self.env.save_state()

        end = time.time()
        print(f'\nRun time: {str(round((end - start) / 60))} minutes \n')

        file = open(self.config.experiment_name + '/neuroended', 'w')
        file.close()

        self.env.state_to_log()

if __name__ == '__main__':
    opt = OptimizationSpecialist()
    opt.run()










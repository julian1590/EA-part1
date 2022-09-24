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
        self.n_vars = (self.env.get_num_sensors() + 1) * self.config.n_hidden_neurons + (self.config.n_hidden_neurons + 1) * 5

    def load_experiment(self):
        # initializes population loading old solutions or generating new ones
        if not os.path.exists(self.config.experiment_name + '/evoman_solstate'):
            print('\nNEW EVOLUTION\n')
            pop = np.random.uniform(self.config.lower_limit,
                                    self.config.upp_limit,
                                    (self.config.n_pop, self.config.n_vars))
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

    def tournament(self, pop):
        c1 = np.random.randint(0, pop.shape[0], 1)
        c2 = np.random.randint(0, pop.shape[0], 1)
        if self.fit_pop[c1] > self.fit_pop[c2]:
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


    def kPointCrossover(self, ind1, ind2, k):
        size = min(len(ind1), len(ind2))
        if k > size:
            k = size
        cpoints = []
        while (len(cxpoints) < k):
            num = np.random.randint(0, size)
            if num not in cxpoints:
                cxpoints.append(num)
        cxpoints.sort()
        for i in range(len(cxpoints)):
            p1 = cxpoints[i]
            ind1[p1::], ind2[p1::] = ind2[p1::], ind1[p1::]
        return ind1, ind2


    def mutGuass(self, offsp, indpb=0.5):
        size = len(offsp)
        for i, m, s in zip(range(size), self.config.mu, self.config.sigma):
            if np.random.random() < indpb:
                offsp[i] += random.gauss(self.config.mu, self.config.sigma)
        return offsp,


    def mutBitFlip(self, ind, indpb):
        size = len(ind)
        for i in range(size):
            if np.random.random() < indpb:
                ind[i] = 1 - ind[i]
        return ind,

    # crossover
    def kissland(self, pop):
        total_offspring = np.zeros((0, self.n_vars))
        for p in range(0, pop.shape[0], 2):
            p1 = self.tournament(pop)
            p2 = self.tournament(pop)

            n_offspring = np.random.randint(1, 3 + 1, 1)[0]
            offspring = np.zeros((n_offspring, self.n_vars))

            for f in range(0, n_offspring):

                cross_prop = np.random.uniform(0, 1)
                # offspring[f] = p1*cross_prop+p2*(1-cross_prop) (original)
                # p1, p2 = twoPointCrossover(p1, p2)
                p1, p2 = self.kPointCrossover(p1, p2, 15)
                offspring[f] = p1 + p2
                if self.config.mutation_algorithm == "gauss":
                    total_offspring  = self.mutGuass(offspring, indpb=0.5)
                elif self.config.mutation_algorithm == "bitflip":
                    total_offspring = self.mutBitFlip(offspring, indpb=0.5)

        return total_offspring

    # kills the worst genomes, and replace with new best/random solutions
    def purge(self, pop, fit_pop):
        worst = int(self.config.n_pop / 4)  # a quarter of the population
        order = np.argsort(fit_pop)
        orderasc = order[0:worst]
        for o in orderasc:
            for j in range(0, self.n_vars):
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
                offspring = self.kissland(pop)
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
                    file_aux = open(self.config.experiment_name + '/results.txt', 'a')
                file_aux.write('\n purge')
                file_aux.close()

                pop, fit_pop = self.purge(pop, fit_pop)
                not_improved = 0

                best = np.argmax(fit_pop)
                std = np.std(fit_pop)
                mean = np.mean(fit_pop)

                # saves results
                file_aux = open(self.config.experiment_name + '/results.txt', 'a')
                print('\n GENERATION ' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
                        round(std, 6)))
                file_aux.write('\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
                    round(std, 6)))
                file_aux.close()

                # saves generation number
                file_aux = open(self.config.experiment_name + '/gen.txt', 'w')
                file_aux.write(str(i))
                file_aux.close()

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

            self.env.state_to_log()  # checks environment state

if __name__ == '__main__':
    opt = OptimizationSpecialist()
    opt.run()










import json
import sys
sys.path.insert(0, 'evoman')
# imports other libs
import time, random
import numpy as np
from collections import defaultdict
from scipy.stats import cauchy
from math import fabs,sqrt
import os
from EA_config import EAConfig


class OptimizationEA:
	def __init__(self):
		self.config = EAConfig()

	def load_experiment(self):
		# initializes population loading old solutions or generating new ones
		if not os.path.exists(self.config.experiment_name + '/evoman_solstate'):
			print('\nNEW EVOLUTION\n')
			pop = np.random.uniform(self.config.lower_limit,
									self.config.upp_limit,
									(self.config.n_pop, self.n_weights))
			fit_pop = self.evaluate(pop)
			age_pop = np.zeros(self.config.n_pop)
			best = np.argmax(fit_pop)
			mean = np.mean(fit_pop)
			std = np.std(fit_pop)
			ini_g = 0
			solutions = [pop, fit_pop]
			self.env.update_solutions(solutions)
		else:
			print('\nCONTINUING EVOLUTION\n')
			self.env.load_state()
			pop, fit_pop, age_pop = self.env.solutions
			best = np.argmax(fit_pop)
			mean = np.mean(fit_pop)
			std = np.std(fit_pop)
			# finds last generation number
			with open(self.config.experiment_name + '/gen.txt', 'r') as f:
				ini_g = int(f.readline())

		with open(self.config.experiment_name + f'/results.txt', 'a') as f:
			f.write('\n gen | best | mean | std')
			print(
				f'\n GENERATION {str(ini_g)} - {str(round(fit_pop[best], 6))} {str(round(mean, 6))} {str(round(std, 6))}')
			f.write(
				f'\n GENERATION {str(ini_g)} | {str(round(fit_pop[best], 6))} | {str(round(mean, 6))} | {str(round(std, 6))}')

		return pop, fit_pop, age_pop, best, mean, std, ini_g

	def simulation(self, env, x):
		fitness, player_life, enemy_life, time = env.play(pcont=x)
		return fitness

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

	def fitnessProportional(self, pop, fit_pop):
		c1 = np.random.randint(0, pop.shape[0], 1)
		c2 = np.random.randint(0, pop.shape[0], 1)
		sum_fitness = np.sum(fit_pop)
		if (fit_pop[c1] / sum_fitness) > (fit_pop[c2] / sum_fitness):
			return pop[c1][0]
		else:
			return pop[c2][0]

	def fitnessProportionalWindowing(self, pop, fit_pop):
		c1 = np.random.randint(0, pop.shape[0], 1)
		c2 = np.random.randint(0, pop.shape[0], 1)
		min_fitness = np.min(fit_pop)
		if (fit_pop[c1] - min_fitness) > (fit_pop[c2] - min_fitness):
			return pop[c1][0]
		else:
			return pop[c2][0]
	
	def fitnessProportionalSigmaScaling(self, pop, fit_pop, mean_fitness, std_fitness):
		c1 = np.random.randint(0, pop.shape[0], 1)
		c2 = np.random.randint(0, pop.shape[0], 1)
		if (max(fit_pop[c1] - (mean_fitness - 2 * std_fitness), 0)) > (max(fit_pop[c2] - (mean_fitness - 2 * std_fitness), 0)):
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
		size = offsp.shape
		mut_prob = np.random.random(size=size) #mutation probability
		random_gauss = np.random.normal(loc=self.config.mu, scale=self.config.sigma, size=size)
		mut_prob[mut_prob < self.config.mutation_prob] *= 0
		mut_prob[mut_prob > 0] -= (mut_prob[mut_prob > 0] - 1)
		temp = random_gauss * mut_prob
		offsp += temp
		return offsp

	def mutCauchy(self, offsp):
		size = offsp.shape
		mut_prob = np.random.random(size=size)  # mutation probability
		random_cauchy = np.random.standard_cauchy(size=size)
		mut_prob[mut_prob < self.config.mutation_prob] *= 0
		mut_prob[mut_prob > 0] -= (mut_prob[mut_prob > 0] - 1)
		temp = random_cauchy * mut_prob
		offsp += temp
		return offsp

	# crossover
	def kissland(self, pop, fit_pop, mean, std, gen):
		total_offspring = np.zeros((0, self.n_weights))
		for p in range(0, pop.shape[0], 2):
			# Selection
			if self.config.fitness_selection == "tournament":
				parent1, parent2 = self.tournament2(pop, fit_pop)
			elif self.config.fitness_selection == "windowing":
				if gen > int(self.config.generations / 2):
					parent1 = self.fitnessProportionalWindowing(pop, fit_pop)
					parent2 = self.fitnessProportionalWindowing(pop, fit_pop)
				else:
					parent1, parent2 = self.tournament2(pop, fit_pop)
			elif self.config.fitness_selection == "sigma_scaling":
				if gen > int(self.config.generations / 2):
					parent1 = self.fitnessProportionalSigmaScaling(pop, fit_pop, mean, std)
					parent2 = self.fitnessProportionalSigmaScaling(pop, fit_pop, mean, std)
				else:
					parent1, parent2 = self.tournament2(pop, fit_pop)
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
					mut_offsp = self.mutGuass(offsp)
				elif self.config.mutation_algorithm == "cauchy":
					mut_offsp = self.mutCauchy(offsp)
				elif self.config.mutation_algorithm == "switch":
					# decrease sigma to switch to exploitation after 15 gens
					if gen > self.config.generations/2:
						mut_offsp = self.mutGuass(offsp)
					else:
						mut_offsp = self.mutCauchy(offsp)
					# 	self.config.mu = 0.5
					# 	self.config.sigma = 0.2
					# mut_offsp = self.mutGuass(offsp)
				mut_offsp[f] = np.array(list(map(lambda y: self.limits(y), mut_offsp[f])))

				total_offspring = np.vstack((total_offspring, mut_offsp))
		return total_offspring

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

	def default_selection(self, pop, fit_pop, age_pop, best):
		fit_pop_cp = fit_pop
		fit_pop_norm = np.array(list(map(lambda y: self.normalize(y, fit_pop_cp), fit_pop)))
		probs = (fit_pop_norm) / (fit_pop_norm).sum()
		chosen = np.random.choice(pop.shape[0], self.config.n_pop, p=probs, replace=False)
		chosen = np.append(chosen[1:], best)
		return pop[chosen], fit_pop[chosen], age_pop[chosen]

	def create_results_dict(self, n_runs, enemies):
		results_dict = defaultdict(dict)
		for enemy in enemies:
			for run in range(n_runs):
				results_dict[str(enemy)][run] = {"means":[], "maximums":[]}
		return results_dict

	def run(self):
		if self.config.run_mode == 'test':
					bsol = np.loadtxt(self.config.experiment_name + '/best.txt')
					print('\n RUNNING SAVED BEST SOLUTION \n')
					self.env.update_parameter('speed', 'normal')
					self.evaluate([bsol])
					sys.exit(0)

		results = self.create_results_dict(self.config.n_runs, self.config.enemies)
		for enemies in self.config.enemies:
			for run in range(self.config.n_runs):
				start = time.time()
				self.env = self.config.init_environment(enemies, run)
				self.n_weights = (self.env.get_num_sensors() + 1) * self.config.n_hidden_neurons + (self.config.n_hidden_neurons + 1) * 5
				self.env.state_to_log()
				pop, fit_pop, age_pop, best, mean, std, ini_g = self.load_experiment()

				# Add the first population results to the results dict
				results[str(enemies)][run]["means"].append(mean)
				results[str(enemies)][run]["maximums"].append(fit_pop[best])

				last_sol = fit_pop[best]
				not_improved = 0
				for i in range(ini_g + 1, self.config.generations):
					offspring = self.kissland(pop, fit_pop, mean, std, i)
					fit_offspring = self.evaluate(offspring)
					pop = np.vstack((pop, offspring))
					age_pop = np.hstack((age_pop, np.zeros(offspring.shape[0])))
					fit_pop = np.append(fit_pop, fit_offspring)

					best = np.argmax(fit_pop)
					fit_pop[best] = float(self.evaluate(np.array([pop[best]]))[0])
					best_sol = fit_pop[best]

					# selection
					if self.config.selection_algorithm == 'default':
						pop, fit_pop, age_pop = self.default_selection(pop, fit_pop, age_pop, best)
					age_pop += 1

					if best_sol <= last_sol:
						not_improved += 1
					else:
						last_sol = best_sol
						not_improved = 0

					if not_improved >= 7:
						with open(self.config.experiment_name + '/results.txt', 'a') as f:
							f.write('\n purge')
						pop, fit_pop = self.purge(pop, fit_pop)
						not_improved = 0

					best = np.argmax(fit_pop)
					std = np.std(fit_pop)
					mean = np.mean(fit_pop)
					if best_sol != fit_pop[best]:
						print("Best solution got selected out")
						x  =1
					results[str(enemies)][run]["means"].append(mean)
					results[str(enemies)][run]["maximums"].append(fit_pop[best])
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
					solutions = [pop, fit_pop, age_pop]
					self.env.update_solutions(solutions)
					self.env.save_state()

				end = time.time()
				print(f'\nRun time: {str(round((end - start) / 60))} minutes \n')

				file = open(self.config.experiment_name + '/neuroended', 'w')
				file.close()

				self.env.state_to_log()

		with open("final_results.json", "w") as f:
			json.dump(results, f)

if __name__ == '__main__':
	opt = OptimizationEA()
	opt.run()










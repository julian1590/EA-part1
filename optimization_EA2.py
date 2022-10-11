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
from EA_config2 import EAConfig2


class OptimizationEA2:
	def __init__(self):
		self.config = EAConfig2()

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
			pop, fit_pop = self.env.solutions
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

		return pop, fit_pop, best, mean, std, ini_g

	def customFitness(self):
		return self.config.enemy_life_weight * (100 - self.env.get_enemylife()) + self.config.player_life_weight * self.env.get_playerlife() - np.log(self.env.get_time())

	def simulation(self, x):
		fitness, player_life, enemy_life, time = self.env.play(pcont=x)
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
		return np.array(list(map(lambda y: self.simulation(y), x)))

	def fitnessProportionalSigmaScaling(self, pop, fit_pop, mean_fitness, std_fitness):
		c1 = np.random.randint(0, pop.shape[0])
		c2 = np.random.randint(0, pop.shape[0])
		if (max(fit_pop[c1] - (mean_fitness - 2 * std_fitness), 0)) > (max(fit_pop[c2] - (mean_fitness - 2 * std_fitness), 0)):
			return pop[c1], fit_pop[c1]
		else:
			return pop[c2], fit_pop[c2]

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

	def getSuccessfullMutations(self, mut_offsp, p1_fitness, p2_fitness):
		fit_offspring = self.evaluate(mut_offsp)
		successfull_mutations = np.where(fit_offspring > max(p1_fitness, p2_fitness), 1, 0)
		return sum(successfull_mutations), len(successfull_mutations), fit_offspring

	# crossover
	def kissland(self, pop, fit_pop, mean, std, gen):
		total_offspring = np.zeros((0, self.n_weights))
		fit_offspring = np.array([])
		successful_mutation = 0
		total_mutations = 0
		for _ in range(0, pop.shape[0], 2):
			# Selection
			parent1, p1_fitness = self.fitnessProportionalSigmaScaling(pop, fit_pop, mean, std)
			parent2, p2_fitness = self.fitnessProportionalSigmaScaling(pop, fit_pop, mean, std)
			n_offsp = np.random.randint(1, 4, 1)[0]
			offsp = np.zeros((n_offsp, self.n_weights))
			for f in range(0, n_offsp):
				# Crossover
				child1, child2 = self.kPointCrossover(parent1, parent2)
				if np.random.uniform(0, 1) < 0.5:
					offsp[f] = child1
				else:
					offsp[f] = child2
			# Mutation
			mut_offsp = self.mutGuass(offsp)
			for f in range(0, n_offsp):
				mut_offsp[f] = np.array(list(map(lambda y: self.limits(y), mut_offsp[f])))
			successful, n_mutations, fit_offsp = self.getSuccessfullMutations(mut_offsp, p1_fitness, p2_fitness)
			successful_mutation += successful
			total_mutations += n_mutations
			fit_offspring = np.append(fit_offspring, fit_offsp)
			total_offspring = np.vstack((total_offspring, mut_offsp))
		# Rechenberg ratio
		ratio = successful_mutation / total_mutations
		if ratio < 0.2 and (self.config.sigma - 0.1) >= 0.0:
			self.config.sigma -= 0.1
		else:
			self.config.sigma += 0.1
		return total_offspring, fit_offspring

	def purge(self, pop, fit_pop):
		worst = int(self.config.n_pop / 4)  # a quarter of the population
		order = np.argsort(fit_pop)
		orderasc = order[0:worst]
		for o in orderasc:
			for j in range(0, self.n_weights):
				pro = np.random.uniform(0, 1)
				if np.random.uniform(0, 1) <= pro:
					np.random.uniform(self.config.upp_limit, self.config.lower_limit) # random dna, uniform dist.
				else:
					pop[o][j] = pop[order[-1:]][0][j] # dna from best
			fit_pop[o] = self.evaluate([pop[o]])
		return pop, fit_pop

	def default_selection(self, pop, fit_pop, best):
		fit_pop_cp = fit_pop
		fit_pop_norm = np.array(list(map(lambda y: self.normalize(y, fit_pop_cp), fit_pop)))
		probs = (fit_pop_norm) / (fit_pop_norm).sum()
		chosen = np.random.choice(pop.shape[0], self.config.n_pop, p=probs, replace=False)
		chosen = np.append(chosen[1:], best)
		return pop[chosen], fit_pop[chosen]

	def create_results_dict(self, n_runs, enemies):
		results_dict = defaultdict(dict)
		for enemy in enemies:
			for run in range(n_runs):
				results_dict[str(enemy)][run] = {"means":[], "maximums":[]}
		return results_dict

	def run(self):
		if self.config.run_mode == 'test':
			bsol = np.loadtxt(self.config.test_EA_dir + '/best.txt')
			print('\n RUNNING SAVED BEST SOLUTION \n')
			self.env = self.config.init_environment(self.config.enemies[0], 0)
			self.env.update_parameter('speed', 'normal')
			self.evaluate([bsol])
			sys.exit(0)

		results = self.create_results_dict(self.config.n_runs, self.config.enemies)
		for enemies in self.config.enemies:
			for run in range(self.config.n_runs):
				start = time.time()
				self.env = self.config.init_environment(enemies, run)
				# Set the fitness function
				self.env.fitness_single = self.customFitness
				self.n_weights = (self.env.get_num_sensors() + 1) * self.config.n_hidden_neurons + (self.config.n_hidden_neurons + 1) * 5
				self.env.state_to_log()
				pop, fit_pop, best, mean, std, ini_g = self.load_experiment()

				# Add the first population results to the results dict
				results[str(enemies)][run]["means"].append(mean)
				results[str(enemies)][run]["maximums"].append(fit_pop[best])

				last_sol = fit_pop[best]
				not_improved = 0
				for i in range(ini_g + 1, self.config.generations):
					offspring, fit_offspring = self.kissland(pop, fit_pop, mean, std, i)
					pop = np.vstack((pop, offspring))
					fit_pop = np.append(fit_pop, fit_offspring)

					best = np.argmax(fit_pop)
					fit_pop[best] = float(self.evaluate(np.array([pop[best]]))[0])
					best_sol = fit_pop[best]

					# selection
					if self.config.selection_algorithm == 'default':
						pop, fit_pop = self.default_selection(pop, fit_pop, best)

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
					solutions = [pop, fit_pop]
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
	opt = OptimizationEA2()
	opt.run()










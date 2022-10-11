import json
import sys
import os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

class EAConfig2:
    def __init__(self):
        with open("EA_config2.json") as f:
            self.params = json.load(f)
        self.n_hidden_neurons = self.params["n_hidden_neurons"]
        self.enemies = self.params["enemies"]
        self.playermode = self.params["playermode"]
        self.enemymode = self.params["enemymode"]
        self.level = self.params["level"]
        self.speed = self.params["speed"]
        self.headless = self.params["headless"]
        self.multiple_mode = self.params["multiplemode"]
        self.run_mode = self.params["run_mode"]
        self.upp_limit = self.params["dom_u"]
        self.lower_limit =  self.params["dom_l"]
        self.n_pop = self.params["npop"]
        self.generations = self.params["generations"]
        self.mutation_prob = self.params["mutation_prob"]
        self.k_points = self.params["k_points"]
        self.mu = self.params["mu"]
        self.sigma = self.params["sigma"]
        self.mutation_algorithm = self.params["mutation_algorithm"]
        self.crossover_algorithm = self.params["crossover_algorithm"]
        self.tournament_size = self.params["tournament_size"]
        self.selection_algorithm = self.params["selection_algorithm"]
        self.fitness_selection = self.params["fitness_selection"]
        self.n_runs = self.params["n_runs"]
        self.test_EA_dir = self.params["test_EA_dir"]
        self.player_life_weight= self.params["player_life_weight"]
        self.enemy_life_weight= 1.0 - self.player_life_weight

        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

    def init_environment(self, enemy, run):
        self.experiment_name = self.create_experiment_name(enemy, run)
        env = Environment(experiment_name=self.experiment_name,
                               enemies=enemy,
                               playermode=self.playermode,
                               multiplemode=self.multiple_mode,
                               player_controller=player_controller(self.n_hidden_neurons),
                               enemymode=self.enemymode,
                               level=self.level,
                               speed=self.speed)
        return env

    def create_experiment_name(self, enemy, run):
        experiment_name = f"EA2_enemy_{enemy}_PlayerWeight-{self.player_life_weight}_fitSelect-{self.fitness_selection}_experiment_pop-{self.n_pop}_tourn_size-{self.tournament_size}_selec-{self.selection_algorithm}_cross-{self.crossover_algorithm}_mut-{self.mutation_algorithm}_mutProb-{self.mutation_prob}"
        if self.mutation_algorithm == "gauss":
            experiment_name += f"_mu-{self.mu}_sigma-{self.sigma}"
        if self.crossover_algorithm == "k_point":
            experiment_name += f"_k-{self.k_points}"
        experiment_name += f"_run-{run}"
        # choose this for not using visuals and thus making experiments faster
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        return experiment_name

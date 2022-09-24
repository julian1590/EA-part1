import json
import sys
import os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

class SpecialistConfig:
    def __init__(self):
        with open("specialist_config.json") as f:
            self.params = json.load(f)
        self.n_hidden_neurons = self.params["n_hidden_neurons"]
        self.enemies = self.params["enemies"]
        self.playermode = self.params["playermode"]
        self.enemymode = self.params["enemymode"]
        self.level = self.params["level"]
        self.speed = self.params["speed"]
        self.headless = self.params["headless"]
        self.run_mode = self.params["run_mode"]
        self.upp_limit = self.params["dom_u"]
        self.lower_limit =  self.params["dom_l"]
        self.n_pop = self.params["npop"]
        self.generations = self.params["generations"]
        self.mutation_rate = self.params["mutation_rate"]
        self.k_points = self.params["k_points"]
        self.mu = self.params["mu"]
        self.sigma = self.params["sigma"]
        self.mutation_algorithm = self.params["mutation_algorithm"]
        self.experiment_name = self.create_experiment_name(self.params)
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

    def init_environment(self):
        env = Environment(experiment_name=self.params["experiment_name"],
                               enemies=self.params["enemies"],
                               playermode=self.params["playermode"],
                               player_controller=player_controller(self.params["n_hidden_neurons"]),
                               enemymode=self.params["enemymode"],
                               level=self.params["level"],
                               speed=self.params["speed"])
        return env

    def create_experiment_name(self):
        experiment_name = f"experiment_pop-{self.n_pop}_k-{self.k_points}"
        # choose this for not using visuals and thus making experiments faster
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        return experiment_name

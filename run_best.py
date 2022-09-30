import json
import sys
import os

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import os
import numpy as np


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# evaluation
def evaluate(x, env):
    return np.array(list(map(lambda y: simulation(env, y), x)))


def test():
    best_run_path = os.path.join("Cauchy-to-gauss",
                                 "enemy_[1]_experiment_pop-110_tourn_size-2_selec-default_cross-k_point_mut-combined_mutProb-0.5_k-10_run-2")

    env = Environment(experiment_name=best_run_path,
                      enemies=[1],
                      playermode="ai",
                      player_controller=player_controller(10),
                      enemymode="static",
                      level=2,
                      speed="normal")
    bsol = np.loadtxt(best_run_path + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    evaluate([bsol], env)
    sys.exit(0)

if __name__ == "__main__":
    test()

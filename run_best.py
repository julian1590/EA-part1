import json
import sys
import os

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import os
import numpy as np

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def get_best(en, run):
    enemy_dir = f"Enemy{en}"
    experiment_path = ""
    for dir in os.listdir(enemy_dir):
        if dir.endswith(f"run-{run}"):
            experiment_path = dir
    best_run_path = os.path.join(enemy_dir, experiment_path)
    return best_run_path


def test():
    experiment_name = "Collecting-boxplots-data"
    enemies = {}
    for en in range(1, 4):
        enemies[en] = {}
        for run in range(10):
            best_run_path = get_best(en, run)
            env = Environment(experiment_name=experiment_name,
                              playermode="ai",
                              enemies = [en],
                              player_controller=player_controller(10),
                              speed="fastest",
                              logs="off",
                              enemymode="static",
                              level = 2)
            bsol = np.loadtxt(best_run_path + '/best.txt')
            individual_gains = []
            for i in range(5):
                print(f"Running enemy {en} run {run} test {i}")
                f, p, e, t = env.play(bsol)
                individual_gain = p - e
                individual_gains.append(individual_gain)
            enemies[en][run] = individual_gains
    with open("boxplot_results.txt", "w") as f:
        json.dump(enemies, f)


if __name__ == "__main__":
    test()

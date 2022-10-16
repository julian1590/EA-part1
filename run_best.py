import json
import sys
import os
from tqdm import tqdm

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import os
import numpy as np

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def get_best(en, EA, run):
    best_run_path = f"experiments_{en}/EA{EA}_enemy_{en}_run-{run}"
    return best_run_path


def test():
    experiment_name = "Collecting-boxplots-data"
    EA = "2"
    train_enemies = [[7, 8], [3, 7, 8]]
    enemies = {}
    for en in train_enemies:
        enemies[str(en)] = {}
        for run in range(10):
            best_run_path = get_best(en, EA, run)
            env = Environment(experiment_name=experiment_name,
                              playermode="ai",
                              enemies = [1, 2, 3, 4, 5, 6, 7, 8],
                              player_controller=player_controller(10),
                              speed="fastest",
                              logs="off",
                              enemymode="static",
                              multiplemode="yes",
                              level = 2)
            bsol = np.loadtxt(best_run_path + '/best.txt')
            individual_gains = []
            for i in range(5):
                print(f"Testing EA {EA} trained on enemies {en} run {run} test {i}")
                f, p, e, t = env.play(bsol)
                individual_gain = p - e
                individual_gains.append(individual_gain)
            enemies[str(en)][run] = individual_gains
    with open(f"boxplot_results_EA-{EA}.json", "w") as f:
        json.dump(enemies, f)


if __name__ == "__main__":
    test()

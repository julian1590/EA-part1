import os
import json
import numpy as np

def get_scores(single_eval):
    fitnesses = []
    for eval in single_eval:
        fitness_line = eval.split(";")[1]
        fitness = float(fitness_line.split(":")[1])
        fitnesses.append(fitness)
    return fitnesses

def run():
    base_experiment_name = "experiment_[3, 7, 8]/EA2_enemy_[3, 7, 8]_run-"
    enemies = [3, 7, 8]
    results = {str(enemies): {run: {"means": [], "maximums": []} for run in range(10)}}
    for run in range(10):
        experiment_name = base_experiment_name + str(run)
        log_file = os.path.join(experiment_name, "results.txt")
        with open(log_file, "r") as f:
            lines = f.readlines()
        log_lines = [line for line in lines if "GENERATION" in line]
        for line in log_lines:
            best, mean = line.split("|")[1:3]
            results[str(enemies)][run]["means"].append(float(mean))
            results[str(enemies)][run]["maximums"].append(float(best))
    with open(f"final_results_{enemies}.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run()
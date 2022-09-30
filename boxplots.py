import os
import sys
import json
from collections import defaultdict

with open("Cauchy-to-gauss/results_enemy1.json") as f:
    results = json.load(f)

best_runs = {str(enemy):{str(run):[] for run in range(10)} for enemy in range(1, 4)}
for enemy in results:
    for run in results[enemy]:
        for i in range(5):
            best_runs[enemy][run].append(max(results[enemy][run]["maximums"]))
print(best_runs)
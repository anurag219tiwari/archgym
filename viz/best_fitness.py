""" Find the best (lowest) fitness value for each workload & agent and create a bar plot"""

import collections
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from absl import app
from absl import flags

# https://www.google.com
flags.DEFINE_string('results_dir', '../results', 'Path to the results folder.')
flags.DEFINE_string('fitness_vals_file', 'fitness.csv', 'Filename where the runtimes are logged.')
flags.DEFINE_string('architecture', 'DRAMSys', 'Architecture for which the time to completion should be plotted.')
flags.DEFINE_list('agents', ['aco', 'bo', 'rw', 'ga'], 'Agents to find the best fitness for.')

FLAGS = flags.FLAGS


# File Structure:
# Architecture (e.g., Sniper)
# ├── aco
# │     ├── aco_logs
# │     └── aco_trajectories
# ├── bo
# │     ├── bo_logs
# │     └── bo_trajectories
# ├── ga
# │     ├── ga_logs
# │     └── ga_trajectories
# └── rw
#     ├── rw_logs
#     └── rw_trajectories


def main(_):
    best_fitness_vals = collections.defaultdict(lambda: collections.defaultdict(lambda: np.Inf))
    best_config = collections.defaultdict(lambda: collections.defaultdict(list))

    # Iterate through all agents and workloads and find min fitness values
    for agent in FLAGS.agents:
        log_dir = os.path.join(FLAGS.results_dir, FLAGS.architecture, agent, f'{agent}_logs')
        experiments = os.listdir(log_dir)
        for exp in experiments:
            if 'DRAMSys' in FLAGS.architecture:
                workload = exp.split('.')[0]
            elif 'Timeloop' in FLAGS.architecture:
                hparam_str = exp.split('=')[1]
                workload = hparam_str.split('_')[0]
                workload += '_v2' if workload == 'mobilenet' else ''

            if agent == 'aco':
                fitness_vals_path = os.path.join(log_dir, exp, f'{exp}_rewards.csv')
            elif agent == 'ga':
                fitness_vals_path = os.path.join(log_dir, exp, 'Y_history.csv')
            else:
                fitness_vals_path = os.path.join(log_dir, exp, FLAGS.fitness_vals_file)

            if os.path.exists(fitness_vals_path):
                fitness_df = pd.read_csv(fitness_vals_path)
                if agent == "ga":  # GA dataframe contains row numbers as first col so remove it
                    fitness_df = fitness_df.iloc[:, 1:]
                min = fitness_df.dropna().to_numpy().min()
                # print(config)
                # print(fitness_df)
                # print(min)
                # print(fitness_df.idxmin())
                if min <= best_fitness_vals[agent][workload]:
                    best_fitness_vals[agent][workload] = min
                    best_config[agent][workload].append(exp)
            else:
                raise FileNotFoundError(f'Fitness file not found for workload:{workload}, experiment:{exp}')

    # Log the best fitness vals and the configs that obtained those vals to the screen for future
    print(best_fitness_vals)
    print(best_config)
    with open(os.path.join(FLAGS.results_dir, FLAGS.architecture, 'best_configs.json'), 'w') as f:
        json.dump(best_config, f)

    # Find the max fitness value of each workload
    agent_names = list(best_fitness_vals.keys())
    workload_names = list(best_fitness_vals["aco"].keys())
    max_vals = {}
    for workload in workload_names:
        max_vals[workload] = -np.Inf
        for agent in agent_names:
            val = best_fitness_vals[agent][workload]
            if val > max_vals[workload]:
                max_vals[workload] = val

    # Normalize the values by this max val
    for workload in workload_names:
        for agent in agent_names:
            best_fitness_vals[agent][workload] = best_fitness_vals[agent][workload] / max_vals[workload]

        # Store best fitness values of each agent after normalization
    # with respect to max fitness val of each workload
    aco_values = list(best_fitness_vals["aco"].values())
    ga_values = list(best_fitness_vals["ga"].values())
    bo_values = list(best_fitness_vals["bo"].values())
    randomwalk_values = list(best_fitness_vals["rw"].values())

    # Place bars and space along x axis
    X_axis = np.arange(len(workload_names))
    plt.bar(X_axis - 0.4, aco_values, 0.2, label='aco')
    plt.bar(X_axis - 0.2, randomwalk_values, 0.2, label='rw')
    plt.bar(X_axis + 0.0, ga_values, 0.2, label='ga')
    plt.bar(X_axis + 0.2, bo_values, 0.2, label='bo')

    # "Prettify" plot with labels, etc. and save fig
    plt.xticks(X_axis, workload_names)
    plt.xlabel("Workloads")
    plt.ylabel("Best Fitness Value")
    plt.title(f'Best Fitness: {FLAGS.architecture}')
    plt.legend()
    plt.savefig("./best_fitness_vals-" + FLAGS.architecture)
    plt.show()


if __name__ == '__main__':
    app.run(main)

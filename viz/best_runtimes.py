import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from absl import app
from absl import flags

flags.DEFINE_string('results_dir', '../results', 'Path to the results folder.')
flags.DEFINE_string('architecture', 'DRAMSys', 'Architecture to process best_runtimes for.')
flags.DEFINE_list('algos', ['aco', 'bo', 'rw', 'ga'], 'Algorithms to find the best runtimes for')

FLAGS = flags.FLAGS

# get the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# config.json file contains the experiment identifier that achieved the best value

exp_identifier = os.path.join(root_dir, "results/dramsys/configs_cleaned.json")
ga_runtimes_file = os.path.join(root_dir, "results/dramsys/time_to_complete_ga.txt")
aco_runtimes = os.path.join(root_dir, "results/dramsys/time_to_complete_aco.txt")
bo_runtimes = os.path.join(root_dir, "results/dramsys/time_to_complete_bo.txt")
rw_runtimes = os.path.join(root_dir, "results/dramsys/time_to_complete_rw.txt")



def rename_keys(algo, key):
    key = key.split("_")
    if algo == "ga":
        new_key = ["ga", key[0], key[-1], key[3], key[6]]
        new_key = "_".join(new_key)
    elif algo == "aco":
        new_key = ["aco", key[0], key[7], key[3], key[5], key[-1]]
        new_key = "_".join(new_key)

    return new_key


def convert_keys(algo, keys_list):
    """
    Convert the keys in the list to string
    """
    keys = []
    for each_key in keys_list:
        each_key = str(each_key)
        keys.append(rename_keys(algo, each_key))

    return keys


def read_runtimes(runtimes_file):
    """
    Read the runtimes from the file
    """
    with open(runtimes_file, "r") as f:
        lines = f.readlines()
    return lines


def find_lowest_matching_runtimes(runtimes, exp_identifier):
    """
    Find the matching runtimes for the experiment identifier
    """
    for runtime in runtimes:
        if exp_identifier in runtime:
            return float(runtime.split(":")[1].strip())


def process_configs(exp_configs):
    stats = []
    for algo in FLAGS.algos:
        if algo not in exp_configs:
            raise ValueError(f'{algo} is not within the best_config dictionary')


    for each_algo in exp_configs:
        ga_runtimes = read_runtimes(ga_runtimes_file)
        if (each_algo == "ga"):
            for each_workload in exp_configs[each_algo]:
                if (each_workload == "cloud-2"):
                    lowest_runtime = 1.0 * 1e19
                    exp_id = ''
                    new_keys = convert_keys(each_algo, exp_configs[each_algo][each_workload])
                    for each_key in new_keys:
                        runtimes = find_lowest_matching_runtimes(ga_runtimes, each_key)
                        if (runtimes < lowest_runtime):
                            lowest_runtime = runtimes
                            exp_id = each_key

                    stats.append(
                        {"algo": each_algo, "workload": each_workload, "runtime": lowest_runtime, "exp_id": exp_id})
                if (each_workload == "cloud-1"):
                    lowest_runtime = 1.0 * 1e19
                    exp_id = ''
                    new_keys = convert_keys(each_algo, exp_configs[each_algo][each_workload])
                    for each_key in new_keys:
                        runtimes = find_lowest_matching_runtimes(ga_runtimes, each_key)
                        if (runtimes < lowest_runtime):
                            lowest_runtime = runtimes
                            exp_id = each_key
                    stats.append(
                        {"algo": each_algo, "workload": each_workload, "runtime": lowest_runtime, "exp_id": exp_id})
                if (each_workload == "stream"):
                    lowest_runtime = 1.0 * 1e19
                    exp_id = ''
                    new_keys = convert_keys(each_algo, exp_configs[each_algo][each_workload])
                    for each_key in new_keys:
                        runtimes = find_lowest_matching_runtimes(ga_runtimes, each_key)
                        if (runtimes < lowest_runtime):
                            lowest_runtime = runtimes
                            exp_id = each_key
                    stats.append(
                        {"algo": each_algo, "workload": each_workload, "runtime": lowest_runtime, "exp_id": exp_id})
                if (each_workload == "random"):
                    lowest_runtime = 1.0 * 1e19
                    exp_id = ''
                    new_keys = convert_keys(each_algo, exp_configs[each_algo][each_workload])
                    for each_key in new_keys:
                        runtimes = find_lowest_matching_runtimes(ga_runtimes, each_key)
                        if (runtimes < lowest_runtime):
                            lowest_runtime = runtimes
                            exp_id = each_key

                    stats.append(
                        {"algo": each_algo, "workload": each_workload, "runtime": lowest_runtime, "exp_id": exp_id})
        elif (each_algo == "aco"):
            runtimes = read_runtimes(aco_runtimes)
            for each_workload in exp_configs[each_algo]:
                if (each_workload == "cloud-2"):
                    lowest_runtime = 1.0 * 1e19
                    exp_id = ''
                    new_keys = convert_keys(each_algo, exp_configs[each_algo][each_workload])
                    for each_key in new_keys:
                        aco_runtimes = find_lowest_matching_runtimes(runtimes, each_key)
                        if (aco_runtimes < lowest_runtime):
                            lowest_runtime = aco_runtimes
                            exp_id = each_key

                    stats.append(
                        {"algo": each_algo, "workload": each_workload, "runtime": lowest_runtime, "exp_id": exp_id})
                if (each_workload == "cloud-1"):
                    lowest_runtime = 1.0 * 1e19
                    exp_id = ''
                    new_keys = convert_keys(each_algo, exp_configs[each_algo][each_workload])
                    for each_key in new_keys:
                        ga_runtimes = find_lowest_matching_runtimes(runtimes, each_key)
                        if (ga_runtimes < lowest_runtime):
                            lowest_runtime = ga_runtimes
                            exp_id = each_key
                    stats.append(
                        {"algo": each_algo, "workload": each_workload, "runtime": lowest_runtime, "exp_id": exp_id})
                if (each_workload == "stream"):
                    lowest_runtime = 1.0 * 1e19
                    exp_id = ''
                    new_keys = convert_keys(each_algo, exp_configs[each_algo][each_workload])
                    for each_key in new_keys:
                        ga_runtimes = find_lowest_matching_runtimes(runtimes, each_key)
                        if (ga_runtimes < lowest_runtime):
                            lowest_runtime = ga_runtimes
                            exp_id = each_key
                    stats.append(
                        {"algo": each_algo, "workload": each_workload, "runtime": lowest_runtime, "exp_id": exp_id})
                if (each_workload == "random"):
                    lowest_runtime = 1.0 * 1e19
                    exp_id = ''
                    new_keys = convert_keys(each_algo, exp_configs[each_algo][each_workload])
                    for each_key in new_keys:
                        ga_runtimes = find_lowest_matching_runtimes(runtimes, each_key)
                        if (ga_runtimes < lowest_runtime):
                            lowest_runtime = ga_runtimes
                            exp_id = each_key

                    stats.append(
                        {"algo": each_algo, "workload": each_workload, "runtime": lowest_runtime, "exp_id": exp_id})
    print(stats)


def main(_):
    # load the config.json file
    with open(exp_identifier) as f:
        exp_configs = json.load(f)

        process_configs(exp_configs=exp_configs)


if __name__ == '__main__':
    app.run(main)

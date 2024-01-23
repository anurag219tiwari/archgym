# File Structure:
# Architecture (e.g., Sniper)
# ├── aco
# │     ├── aco_logs
# │     └── time_to_complete_aco.txt
# │     └── aco_trajectories
# ├── bo
# │     ├── bo_logs
# │     └── time_to_complete_bo.txt
# │     └── bo_trajectories
# ├── ga
# │     ├── ga_logs
# │     └── time_to_complete_ga.txt
# │     └── ga_trajectories
# └── rw
#     ├── rw_logs
#     └── time_to_complete_rw.txt
#     └── rw_trajectories

from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import collections
import pandas as pd
import json
import os

flags.DEFINE_string('results_dir', '../results', 'Path to the results folder.')
flags.DEFINE_string('bf_filename', 'configs_cleaned.json', 'Filename of the configs for the best fitness file.')
flags.DEFINE_string('architecture', 'DRAMSys', 'Architecture for which the time to completion should be plotted.')
flags.DEFINE_float('bar_width', 1.0, 'Width of each bar plot.')

FLAGS = flags.FLAGS


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height.
    https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
    """
    for container in rects:
        for rect in container:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.0 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')


def plot_bar_charts(data):
    # fig, ax = plt.subplots(figsize=(15, 20))
    fig, ax = plt.subplots()

    # Space the bars appropriately
    x = []
    for i in range(len(data.keys())):
        x.append(FLAGS.bar_width * i + (FLAGS.bar_width / 4 * i))
    x = np.array(x)
    print(x)
    labels = list(data.keys())

    plot_data = collections.defaultdict(list)
    for algo in labels:
        for workload, values in data[algo].items():
            data_point = min(values)
            plot_data[workload].append(data_point)  # aggregate values

    # Create rectangles based on plot_data
    rectime_to_completets = []
    start_pos = FLAGS.bar_width / 2
    for workload, values in plot_data.items():
        rects.append(ax.bar(x - start_pos, values, width=FLAGS.bar_width / 4, label=workload, align='edge'))
        start_pos -= FLAGS.bar_width / 4
    autolabel(rects=rects, ax=ax)
    ax.set_ylabel('Time to Completion')
    ax.set_xlabel('Algorithms')

    ax.set_title(f'Time to Completion: {FLAGS.architecture}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title='Workloads')

    fig.tight_layout()

    plt.savefig(f'./best_time_to_complete-{FLAGS.architecture}.png')
    plt.show(aspect='auto')


def main(_):
    # Load the best fitness data
    with open(os.path.join(FLAGS.results_dir, FLAGS.architecture, FLAGS.bf_filename), 'r') as fitness_file:
        best_fitness_data = json.load(fitness_file)

    # Load the "time to completion" files
    algo_file_dict = {algo: os.path.join(FLAGS.results_dir, FLAGS.architecture, algo,
                                         'time_to_complete_rw.txt' if algo == 'randomwalker' else f'time_to_complete_{algo}.txt')
                      for algo in best_fitness_data}
    algo_df_dict = collections.defaultdict()
    for algo, v in algo_file_dict.items():
        with open(v, 'r') as ttc_file:
            algo_df_dict[algo] = pd.read_csv(ttc_file, header=None, sep=' ')
            algo_df_dict[algo] = dict(zip(algo_df_dict[algo][0], algo_df_dict[algo][1]))

    # Match the experiment name with the time to completion
    value_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for algo in algo_df_dict:
        for workload, experiments in best_fitness_data[algo].items():
            for exp in experiments:
                if 'DRAMSys' in FLAGS.architecture:
                    hparam_text = exp.split('.', 1)[1]
                    hparam_text = hparam_text.split('_')
                elif 'Timeloop' in FLAGS.architecture:
                    hparam_text = exp.split('_')[2:]
                    hparam_text = [x.split('=')[1] for x in hparam_text if x and 'v2' not in x]

                # Different algorithms have different naming schemes
                if algo == 'ga':
                    if 'DRAMSys' in FLAGS.architecture:
                        num_iter, num_agents, prob_mut = hparam_text[3], hparam_text[6], hparam_text[9]
                        lookup_key = f'{algo}_{workload}.stl_{prob_mut}_{num_iter}_{num_agents}:'
                    elif 'Timeloop' in FLAGS.architecture:
                        timesteps, num_iter, prob_mut, num_agents = hparam_text[0], hparam_text[1], hparam_text[2], \
                                                                    hparam_text[3]
                        lookup_key = f'{algo}_wl={workload}_t={timesteps}_ms={num_iter}_pm={prob_mut}_ps={num_agents}_:'

                elif algo == 'aco':
                    if 'DRAMSys' in FLAGS.architecture:
                        ant_count, greediness, evaporation, depth = hparam_text[3], hparam_text[5], hparam_text[7], \
                                                                    hparam_text[9]
                        lookup_key = f'{algo}_{workload}.stl_{evaporation}_{ant_count}_{greediness}_{depth}:'
                    elif 'Timeloop' in FLAGS.architecture:
                        timesteps, ant_count, greediness, evaporation, depth = hparam_text[0], hparam_text[1], \
                                                                               hparam_text[2], hparam_text[3], \
                                                                               hparam_text[4]

                        lookup_key = f'{algo}_wl={workload}_t={timesteps}_na={ant_count}_g={greediness}_ev={evaporation}_dpt={depth}_:'

                elif algo == 'rw':
                    if 'DRAMSys' in FLAGS.architecture:
                        num_steps = hparam_text[3]
                        lookup_key = f'random_walk_{workload}.stl_{num_steps}:'
                    elif 'Timeloop' in FLAGS.architecture:
                        timesteps, num_steps = hparam_text[0], hparam_text[1]
                        lookup_key = f'{algo}_wl={workload}_t={timesteps}_ms={num_steps}_:'
                elif algo == 'bo':
                    if 'DRAMSys' in FLAGS.architecture:
                        random_state, num_iter = hparam_text[3], hparam_text[6]
                        lookup_key = f'{algo}_{workload}.stl_{random_state}_{num_iter}:'
                    elif 'Timeloop' in FLAGS.architecture:
                        timesteps, num_iter, random_state = hparam_text
                        lookup_key = f'{algo}_wl={workload}_t={timesteps}_ms={num_iter}_sd={random_state}_:'

                else:
                    raise NotImplementedError('Only ga, randomwalker, bo, and aco are supported')

                value_dict[algo][workload].append(algo_df_dict[algo][lookup_key])

    plot_bar_charts(value_dict)


if __name__ == '__main__':
    app.run(main)

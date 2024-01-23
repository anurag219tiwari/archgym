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
flags.DEFINE_string('architecture', 'DRAMSys', 'Architecture for which the time to completion should be plotted.')
flags.DEFINE_list('agents', ['aco', 'bo', 'random_walk', 'ga'], 'Agents to find the best fitness for.')
flags.DEFINE_list('workloads', ['stream', 'cloud-1', 'cloud-2', 'random'], 'Agents to find the best fitness for.')
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
    err_data = collections.defaultdict(list)
    for algo in labels:
        for workload, values in data[algo].items():
            data_point, err = values[0][0], values[0][1]
            plot_data[workload].append(data_point)  # aggregate values
            # divide by 2 due to matplotlib bar https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html
            err_data[workload].append(err / 2)

            # Create rectangles based on plot_data
    rects = []
    start_pos = FLAGS.bar_width / 2
    for workload, values in plot_data.items():
        rects.append(ax.bar(x - start_pos, values, yerr=err_data[workload], capsize=5, ecolor='black',
                            width=FLAGS.bar_width / 4, label=workload,
                            align='edge'))
        start_pos -= FLAGS.bar_width / 4
    autolabel(rects=rects, ax=ax)
    ax.set_ylabel('Time to Completion')
    ax.set_xlabel('Algorithms')

    ax.set_title(f'All Time to Completion: {FLAGS.architecture}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title='Workloads')
    ax.yaxis.grid(True)

    fig.tight_layout()

    plt.savefig(f'./all_time_to_complete-{FLAGS.architecture}.png')
    plt.show(aspect='auto')


def main(_):
    # Load the "time to completion" file
    ttc_path = os.path.join(FLAGS.results_dir, FLAGS.architecture, 'time_to_complete.txt')

    with open(ttc_path, 'r') as ttc_file:
        df = pd.read_csv(ttc_file, header=None, sep=' ')

    # Match the experiment name with the time to completion
    value_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for agent in FLAGS.agents:
        agent_df = df.where(df[0].str.contains(agent)).dropna()
        print(f'{len(agent_df)} exps in {agent}')
        for workload in FLAGS.workloads:
            wl_df = agent_df.where(agent_df[0].str.contains(workload)).dropna()
            value_dict[agent][workload].append((wl_df[1].mean(), wl_df[1].std()))

    #
    # for algo in algo_df_dict:
    #     df = algo_df_dict[algo]
    #     for workload in FLAGS.workloads:
    #         filtered_df = df.where(df[0].str.contains(workload)).dropna()
    #         value_dict[algo][workload].append((filtered_df[1].mean(), filtered_df[1].std()))
    #         del filtered_df

    plot_bar_charts(value_dict)


if __name__ == '__main__':
    app.run(main)

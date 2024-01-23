"""Visualize a specified trajectory from ArchGym.
Example expected directory structure:
├── DRAMSys_hard
│         ├── aco
│         │         ├── aco_logs
│         │         ├── aco_trajectories
│         │         ├── time_to_complete_aco.txt
│         ├── best_configs.json
│         ├── bo
│         │         ├── bo_logs
│         │         ├── bo_trajectories
│         │         ├── time_to_complete_bo.txt
│         ├── ga
│         │         ├── ga_logs
│         │         ├── ga_trajectories
│         │         ├── time_to_complete_ga.txt
│         └── rw
│             ├── rw_logs
│             ├── rw_trajectories
│             ├── time_to_complete_rw.txt
"""
import collections
import os
import matplotlib.pyplot as plt
from absl import app
from absl import flags
from envlogger import reader
import json
import numpy as np
import sys
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

FLAGS = flags.FLAGS
flags.DEFINE_string('results_dir', '../results', 'Path to the results folder.')
flags.DEFINE_string('save_dir', './', 'Path to folder for saving the plots.')
flags.DEFINE_string('architecture', 'DRAMSys_hard', 'Architecture for which the time to completion should be plotted.')
flags.DEFINE_string('agent', 'ga', 'Agent to visualize.')
flags.DEFINE_string('best_config_json', 'best_configs.json', 'JSON File with Best Configs')
flags.DEFINE_string('workload', '', 'Workload of the trajectory to visualize.')
flags.DEFINE_string('hparam_str', '', 'String of hyperparameters that appear after the workload. '
                                      'E.g, `stl_ant_count_16_greediness_0.0_evaporation_0.1_depth_2` in'
                                      ' cloud-1.stl_ant_count_16_greediness_0.0_evaporation_0.1_depth_2  ')


def get_best_file_names_for_agent(agent):
    '''Reads and returns the best_config_json file
    Returns:
        best_config_files_json: A json file containing best config file names
    '''
    best_config_files_json = {}
    try:
        # Read in the best configuration file names produced by best_fitness.py
        with open(os.path.join(FLAGS.results_dir, FLAGS.architecture, FLAGS.best_config_json), 'r') as best_config_file:
            best_config_files_json = json.load(best_config_file)
    except:
        raise FileNotFoundError(f'Best Config json file {FLAGS.best_config_json} does not exist')
    return best_config_files_json


def main(_):
    if FLAGS.agent not in ('aco', 'rw', 'ga', 'bo'):
        raise ValueError(f'{FLAGS.agent} is not supported')
    best_config_files_json = get_best_file_names_for_agent(FLAGS.agent)
    # Iterate through all best workloads of agent

    min_energy = np.inf
    min_action = None
    for workload in best_config_files_json[FLAGS.agent].keys():
        FLAGS.workload = workload
        # Iterate through all best hyperparameters for each workload of agent
        for workload_and_hparam_str in best_config_files_json[FLAGS.agent][workload][:10]:
            # Assume "workload."" will only show up at the first part of the data directory name
            FLAGS.hparam_str = workload_and_hparam_str.replace(workload + '.', '')
            # Construct the directory of data
            if 'DRAMSys' in FLAGS.architecture:
                data_dir = os.path.join(FLAGS.results_dir, FLAGS.architecture, FLAGS.agent,
                                        f'{FLAGS.agent}_trajectories',
                                        f'{FLAGS.workload}.{FLAGS.hparam_str}')
            else:
                data_dir = os.path.join(FLAGS.results_dir, FLAGS.architecture, FLAGS.agent,
                                        f'{FLAGS.agent}_trajectories',
                                        f'{FLAGS.agent}_wl={FLAGS.workload}{FLAGS.hparam_str}')
            if os.path.exists(data_dir):
                # Read in data
                with reader.Reader(data_directory=data_dir) as r:
                    episodes = [e for e in r.episodes]
                    if FLAGS.agent == 'aco':
                        if 'DRAMSys' in FLAGS.architecture:
                            ant_count = int(FLAGS.hparam_str.split('_')[3])
                            # ant_data = [collections.defaultdict(list) for _ in range(ant_count)]
                            # Skip the first ant since it does a random walk
                            for i, e in enumerate(episodes[1:]):
                                steps = list(e)
                                assert len(steps) == 2
                                # for position, metric in enumerate(('Energy', 'Power', 'Latency')):
                                for position, metric in enumerate(('Energy',)):
                                    metric_value = steps[1].timestep.observation.flatten()[position]
                                    # ant_data[i % ant_count][metric].append(metric_value)
                                    if metric_value < min_energy:
                                        min_energy = metric_value
                                        min_action = steps[1].action
                        elif 'Timeloop' in FLAGS.architecture:
                            ant_count = int(FLAGS.hparam_str.split('_')[2].split('=')[1])
                            ant_data = [collections.defaultdict(list) for _ in range(ant_count)]
                            for i in range(1, len(episodes), 2):
                                steps = list(episodes[i])
                                assert len(steps) == 2
                                for agent_num in range(len(steps[1].timestep.observation)):
                                    for position, metric in enumerate(('Energy', 'Power', 'Latency')):
                                        ant_data[agent_num][metric].append(
                                            steps[1].timestep.observation[agent_num][position])
                        # Plot trajectory data
                        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        # for i, metric in enumerate(('Energy', 'Power', 'Latency')):
                        #     axs[i].set_title(f'{metric}')
                        #     axs[i].set_ylabel(metric)
                        #     axs[i].set_xlabel('Timesteps')
                        #     for ant in range(ant_count):
                        #         y_data = ant_data[ant][metric]
                        #         x_data = range(1, len(y_data) + 1, 1)
                        #         axs[i].plot(x_data, y_data, 'o', linestyle='solid', label=f'Ant {ant + 1}')
                        #     axs[i].legend()

                    elif FLAGS.agent == 'rw':
                        plot_data = collections.defaultdict(list)
                        timesteps = 0
                        for episode in episodes:
                            ep_timesteps = 0
                            for step in episode[1:]:
                                if 'Timeloop' in FLAGS.architecture:
                                    energy, power, latency = step.timestep.observation
                                else:
                                    energy, power, latency = step.timestep.observation[0]
                                plot_data['Energy'].append(energy)
                                plot_data['Power'].append(power)
                                plot_data['Latency'].append(latency)
                                timesteps += 1
                                ep_timesteps += 1
                                # print(step)
                            # print(f'Episode timesteps: {ep_timesteps}')
                        # Plot trajectory data
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        for i, metric in enumerate(('Energy', 'Power', 'Latency')):
                            y_data = plot_data[metric]
                            x_data = range(len(y_data))
                            axs[i].set_title(f'{metric}')
                            axs[i].set_ylabel(metric)
                            axs[i].set_xlabel('Timesteps')
                            axs[i].plot(x_data, y_data, 'o', linestyle='solid', label=f'RW Agent')
                            axs[i].legend()
                        print(f'{len(episodes)} episodes')
                        print(f'{timesteps} timesteps')
                    elif FLAGS.agent == 'ga':
                        if 'DRAMSys' in FLAGS.architecture:
                            agent_count = int(FLAGS.hparam_str.split('_')[6])
                        elif 'Timeloop' in FLAGS.architecture:
                            agent_count = int(FLAGS.hparam_str.split('_')[4].split('=')[1])
                        agent_data = [collections.defaultdict(list) for _ in range(agent_count)]
                        if 'DRAMSys' in FLAGS.architecture:
                            for i, e in enumerate(episodes[1:]):
                                steps = list(e)
                                assert len(steps) == 2
                                for position, metric in enumerate(('Energy', 'Power', 'Latency')):
                                    agent_data[i % agent_count][metric].append(
                                        steps[1].timestep.observation.flatten()[position])
                        elif 'Timeloop' in FLAGS.architecture:
                            for i in range(0, len(episodes), 2):
                                steps = list(episodes[i])
                                assert len(steps) == 2
                                for agent_num in range(len(steps[1].timestep.observation)):
                                    for position, metric in enumerate(('Energy', 'Power', 'Latency')):
                                        agent_data[agent_num][metric].append(
                                            steps[1].timestep.observation[agent_num][position])
                        # Plot trajectory data
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        for i, metric in enumerate(('Energy', 'Power', 'Latency')):
                            axs[i].set_title(f'{metric}')
                            axs[i].set_ylabel(metric)
                            axs[i].set_xlabel('Timesteps')
                            for agent in range(agent_count):
                                y_data = agent_data[agent][metric]
                                x_data = range(1, len(y_data) + 1, 1)
                                axs[i].plot(x_data, y_data, 'o', linestyle='solid', label=f'Agent {agent + 1}')
                            axs[i].legend()
                    elif FLAGS.agent == 'bo':
                        plot_data = collections.defaultdict(list)
                        for episode in episodes:
                            for step in episode[1:]:
                                if 'Timeloop' in FLAGS.architecture:
                                    energy, power, latency = step.timestep.observation
                                else:
                                    energy, power, latency = step.timestep.observation[0]
                                plot_data['Energy'].append(energy)
                                plot_data['Power'].append(power)
                                plot_data['Latency'].append(latency)
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        for i, metric in enumerate(('Energy', 'Power', 'Latency')):
                            y_data = plot_data[metric]
                            x_data = range(len(y_data))
                            axs[i].set_title(f'{metric}')
                            axs[i].set_ylabel(metric)
                            axs[i].set_xlabel('Timesteps')
                            axs[i].plot(x_data, y_data, 'o', linestyle='solid', label=f'BO Agent')
                            axs[i].legend()
                    # fig.suptitle(f'Metrics for {FLAGS.agent} on {FLAGS.architecture}', fontsize=14)
                    # plt.savefig(
                    #     os.path.join(FLAGS.save_dir, f'./obs_arch_{FLAGS.architecture}_agent_{FLAGS.agent}.png'))
                    # plt.show()
            else:
                # raise FileNotFoundError(f'Trajectory path {data_dir} does not exist')
                print(data_dir)
        break
    # Plot for the agent
    min_action
    plot_action(min_action)


def plot_action(action):
    # categories = list(action.keys())

    categories = ['RefreshMaxPostponed', 'RefreshMaxPulledin', 'MaxActiveTransactions', 'RequestBufferSize']
    categories = [*categories, categories[0]]
    energy_vals = [action[c] for c in categories]

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(energy_vals))

    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    plt.plot(label_loc, energy_vals, label='Energy')
    # plt.plot(label_loc, restaurant_2, label='Restaurant 2')
    # plt.plot(label_loc, restaurant_3, label='Restaurant 3')
    # plt.title('Restaurant comparison', size=20)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    app.run(main)

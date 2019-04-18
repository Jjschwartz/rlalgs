"""
An analyser class that collates data from experimental runs and runs a sime analysis on it.
Providing some simple visualisations and statistics
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def load_experiment_run(data_dir):
    """
    Loads a single experiments data (i.e. for a single seed)

    Assumes standard data output, with one .txt file and one config.json file

    Arguments:
        str data_dir : path to directory containing data

    Returns:
        pd_dataframe data : the experiment data
        dict config : the experiment config information
    """
    data_file = os.path.join(data_dir, "progress.txt")
    config_file = os.path.join(data_dir, "config.json")
    data = pd.read_table(data_file)
    with open(config_file, "r") as fin:
        config = json.load(fin)
    return data, config


def load_all_experiment_runs(exp_parent_dir):
    """
    Loads all experiment runs for a given experiment, where the seeds are diffrent.

    Arguments:
        str exp_parent_dir : the parent directory for experiment

    Returns:
        [pd_dataframe] data : list of data from each experiment
        [dict] config : list of all config info for each experiment
    """
    exp_run_dirs = get_subdirectories(exp_parent_dir)
    data = []
    configs = []
    for run_dir in exp_run_dirs:
        run_data, run_config = load_experiment_run(run_dir)
        data.append(run_data)
        configs.append(run_config)
    return data, configs


def average_over_runs(data):
    """
    Averages epoch data over different runs
    """
    data_concat = pd.concat(data)
    by_row_index = data_concat.groupby(data_concat.index)
    avg_data = by_row_index.mean()
    err_data = by_row_index.std()
    return avg_data, err_data


def plot_data(config, df, x_key, y_key, err_df=None):
    """
    Plots data on 2D plot
    """
    x = df[x_key]
    y = df[y_key]
    err = err_df[y_key]

    plt.plot(x, y)
    if err is not None:
        plt.fill_between(x, y - err, y + err, alpha=0.5)
    plt.xlabel("epoch")
    plt.ylabel("average return")
    plt.title(config["exp_name"])

    plt.show()


def get_subdirectories(parent_dir):
    """ Get all subdirectories of parent """
    sub_dirs = []
    for name in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, name)):
            sub_dirs.append(os.path.join(parent_dir, name))
    return sub_dirs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help="path to experiment parent directory")
    args = parser.parse_args()

    print("\nAnalyser")
    data, configs = load_all_experiment_runs(args.exp_dir)
    avg_data, err_data = average_over_runs(data)
    plot_data(configs[0], avg_data, "epoch", "avg_return", err_data)

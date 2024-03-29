"""
Module for plotting logger output from running experiment

Plots:
- average return per epoch

Reports statistics on:
- return
- episode lengths
- training length

Usage:

    python plot.py path/to/data/file.txt [--smooth n]

"""
import pandas as pd
import matplotlib.pyplot as plt

LINE = "\n" + "-" * 60 + "\n"


def plot(file_path, smooth_period):

    df = pd.read_table(file_path)
    x = df["epoch"]
    y = df["avg_return"]
    # eps = df["total_eps"]
    ep_lens = df["avg_ep_lens"]

    num_epochs = x.count()
    y_smooth = y.rolling(smooth_period, min_periods=1)
    y_smooth_mean = y_smooth.mean()
    y_rolling_std = y_smooth.std()

    print(LINE)
    print("RETURN\n")
    print("All stats are for epochs which are averages of episodic returns\n")
    print("\tTotal return: {:.3f}".format(y.sum()))
    print("\tAverage epoch return: {:.3f}".format(y.mean()))
    print("\tStandard dev: {:.3f}".format(y.std()))
    print("\tAverage epoch return for last 100 epochs: {:.3f}".format(y.tail(100).mean()))
    print("\tStandard dev for last 100 epochs: {:.3f}".format(y.tail(100).std()))
    print("\tMax value: {}".format(y.max()))
    print("\tMax epoch: {}".format(y.idxmax()))
    print("\tMin value: {}".format(y.min()))
    print("\tMin epoch: {}".format(y.idxmin()))

    print("\n\nEPISODES\n")
    print("All stats are for epochs which are averages of episode lengths (except total episodes)\n")
    print("\tNumber of epochs: {}".format(num_epochs))
    # print("\tTotal episodes: {}".format(eps.max()))
    # print("\tAverage episodes per epoch: {}".format(eps.max()/num_epochs))
    print("\tMax episode length: {}".format(ep_lens.max()))
    print("\tMax epoch: {}".format(ep_lens.idxmax()))
    print("\tMin episode length: {}".format(ep_lens.min()))
    print("\tMin epoch: {}".format(ep_lens.idxmin()))
    print("\tAverage episode length: {:.3f}".format(ep_lens.mean()))
    print("\tStandard dev: {:.3f}".format(ep_lens.std()))

    print("\n\nTRAINING TIME\n")
    try:
        epoch_times = df["epoch_time"]
        training_time = epoch_times.sum()
        print("\tTotal training time: {:.3f} secs / {:.3f} mins / {:.3f} hours"
              .format(training_time, training_time/60.0, training_time/(60.0 * 60)))
        print("\tAverage epoch time: {:.3f} secs".format(epoch_times.mean()))
        print("\tStandard dev: {:.3f} secs".format(epoch_times.std()))
    except Exception:
        print("No time information")
    print(LINE)

    plt.plot(x, y_smooth_mean)
    plt.fill_between(x, y_smooth_mean-y_rolling_std, y_smooth_mean+y_rolling_std, alpha=0.5)
    plt.xlabel("epoch")
    plt.ylabel("average return")
    plt.title(file_path.split("/")[-1])

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='file', type=str,
                        help='path to logger output file')
    parser.add_argument('--smooth', type=int, default=1,
                        help='Number of epochs to smooth returns over')
    args = parser.parse_args()

    print("\nResults plotter")
    print("Plotting results from {}".format(args.file))
    print("Using smoothing period of {}".format(args.smooth))
    plot(args.file, args.smooth)

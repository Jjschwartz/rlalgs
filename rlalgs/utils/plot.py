"""
Module for plotting logger output from running experiment

Plots:
- average return per epoch

Usage:

    python plot.py path/to/data/file.txt

"""
import pandas as pd
import matplotlib.pyplot as plt


def plot(file_path):

    df = pd.read_table(file_path)

    x = df["epoch"]
    y = df["avg_return"]

    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.ylabel("average return")
    plt.title(file_path.split("/")[-1])

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='file', type=str,
                        help='path to logger output file')
    args = parser.parse_args()

    print("\nResults plotter")
    plot(args.file)

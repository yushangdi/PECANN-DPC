import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from pathlib import Path
import sys

abspath = Path(__file__).resolve().parent.parent
sys.path.append(str(abspath))
from plotting_utils import dataset_name_map


plt.rcParams.update({"font.size": 14})


def plot_scalability_by_number_of_threads(csv_folder, threads):
    file_pattern = os.path.join(csv_folder, "*_restricted_*.csv")
    csv_files = glob.glob(file_pattern)

    dfs = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        dfs.append(df)

    df = pd.concat(dfs)

    df = df.sort_values(by="num_threads")

    grouped_data = df.groupby("dataset")

    speedups = []
    for name, group in grouped_data:
        group["total_speedup"] = group["Total time"].iloc[0] / group["Total time"]
        speedups.append(group["total_speedup"].to_numpy())
        plt.plot(
            group["num_threads"],
            group["total_speedup"],
            label=f"{dataset_name_map[name]}",
            marker="o",
        )
    speedups = np.array(speedups)
    print(np.mean(speedups, axis=0))

    if threads:
        plt.xlabel("Number of Threads")
        plt.ylabel("Speedup")
        plt.legend(title="Dataset", loc="upper left")

        plt.axvline(30, c="black", linestyle=":")
        plt.text(27, 0.75, "One complete NUMA node", rotation=90)

        plt.axvline(60, c="black", linestyle=":")
        plt.text(57, 0.75, "Two complete NUMA nodes", rotation=90)

        plt.savefig("results/paper/thread_scaling.pdf")
    else:
        plt.xlabel("Number of Cores")
        plt.ylabel("Speedup")
        plt.legend(title="Dataset", loc="upper left")

        plt.axvline(15, c="black", linestyle=":")
        plt.text(13.5, 0.75, "One NUMA node", rotation=90)

        plt.axvline(30, c="black", linestyle=":")
        plt.text(28.5, 0.75, "Two NUMA nodes", rotation=90)

        plt.savefig("results/paper/core_scaling.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot total time vs number of threads for each dataset"
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing the dataset size scaling experiment data",
    )

    parser.add_argument("--threads", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    plot_scalability_by_number_of_threads(args.folder, args.threads)

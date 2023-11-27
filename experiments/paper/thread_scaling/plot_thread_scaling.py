import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


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

    for name, group in grouped_data:
        group["total_speedup"] = group["Total time"].iloc[0] / group["Total time"]

        plt.plot(
            group["num_threads"], group["total_speedup"], label=f"{name}", marker="o"
        )

    if threads:
        plt.title("Clustering Time Speedup vs. Number of Threads")

        plt.xlabel("Number of Threads")
        plt.ylabel("Speedup")
        plt.legend(title="Dataset")

        plt.axvline(30, c="black", linestyle=":")
        plt.text(27.5, 0.75, "One complete NUMA node", rotation=90)

        plt.axvline(60, c="black", linestyle=":")
        plt.text(57.5, 0.75, "Two complete NUMA nodes", rotation=90)

        plt.savefig("results/paper/thread_scaling.pdf")
    else:
        plt.title("Clustering Time Speedup vs. Number of Cores (No Hyperthreading)")

        plt.xlabel("Number of Cores")
        plt.ylabel("Speedup")
        plt.legend(title="Dataset")

        plt.axvline(15, c="black", linestyle=":")
        plt.text(14, 0.75, "One NUMA node", rotation=90)

        plt.axvline(30, c="black", linestyle=":")
        plt.text(29, 0.75, "Two NUMA nodes", rotation=90)

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

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def plot_scalability_by_number_of_threads(csv_folder):
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

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend(title="Dataset")
    plt.savefig("results/paper/thread_scaling.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot total time vs number of threads for each dataset"
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing the dataset size scaling experiment data",
    )

    args = parser.parse_args()

    plot_scalability_by_number_of_threads(args.folder)

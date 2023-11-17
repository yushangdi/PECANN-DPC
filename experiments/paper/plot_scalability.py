import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_scalability_by_dataset_size(csv_file):
    df = pd.read_csv(csv_file)

    grouped_data = df.groupby("num_threads")

    for name, group in grouped_data:
        plt.plot(
            group["dataset"].str[9:].astype("int64"),
            group["Total time"],
            label=f"{name} threads",
        )

    plt.xlabel("Dataset Size")
    plt.ylabel("Total Time")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(title="Dataset")
    plt.savefig("results/paper/synthetic.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot total time vs dataset size for each number of threads"
    )

    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing the scalability experiment data data",
    )

    args = parser.parse_args()

    plot_scalability_by_dataset_size(args.csv_file)

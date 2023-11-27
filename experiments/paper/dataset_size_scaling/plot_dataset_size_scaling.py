import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_scalability_by_dataset_size(csv_file):
    df = pd.read_csv(csv_file)

    df["num_clusters"] = df["dataset"].str.split("_").str[2]
    df["dataset_size"] = df["dataset"].str.split("_").str[1].astype("int64")
    grouped_data = df.groupby("num_clusters")

    for name, group in grouped_data:
        # Linear regression
        x = np.log(group["dataset_size"]).values.reshape(-1, 1)
        y = np.log(group["Total time"]).values
        model = LinearRegression().fit(x, y)
        k = model.coef_[0]

        plt.plot(
            group["dataset_size"],
            group["Total time"],
            label=f"{name} clusters, slope = {k:.2f}",
        )

    plt.title("Clustering Time vs. Dataset Size")
    plt.xlabel("Dataset Size")
    plt.ylabel("Clustering Time (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("results/paper/dataset_size_scaling.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot total time vs dataset size for each number of threads"
    )

    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing the dataset size scaling experiment data",
    )

    args = parser.parse_args()

    plot_scalability_by_dataset_size(args.csv_file)

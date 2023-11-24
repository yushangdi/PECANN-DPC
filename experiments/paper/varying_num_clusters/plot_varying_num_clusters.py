import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_ari_by_cluster_offset(csv_path):
    df = pd.read_csv(csv_path)

    grouped_data = df.groupby("dataset")

    gt_num_clusters = {
        "mnist": 10,
        "imagenet": 1000,
        "arxiv-clustering-s2s": 180,
        "reddit-clustering": 50,
        "birds": 525,
    }
    for name, group in grouped_data:
        group["cluster_ratio"] = group["num_clusters"] / gt_num_clusters[name]

        plt.plot(group["cluster_ratio"], group["ARI"], label=f"{name}")

    plt.title('Effect of Clustering with the "Wrong" Number of Clusters')
    plt.axvline(1, c="black", linestyle=":")
    plt.text(1.1, 0, "Correct number of clusters", rotation=90)
    plt.xlabel("Cluster Ratio: # Clusters Used / # Clusters in Ground Truth")
    plt.ylabel("ARI")
    plt.legend(title="Dataset")
    plt.savefig("results/paper/varying_num_clusters.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ARI vs percent of the ground truth number of clusters"
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the csv containing the varying number of clusters experiment data",
    )

    args = parser.parse_args()

    plot_ari_by_cluster_offset(args.csv_path)

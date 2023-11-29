import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys

abspath = Path(__file__).resolve().parent.parent
sys.path.append(str(abspath))
from plotting_utils import set_superplot_font_sizes, reset_font_sizes, dataset_name_map

plt.rcParams.update({"font.size": 14})

gt_num_clusters = {
    "mnist": 10,
    "imagenet": 1000,
    "arxiv-clustering-s2s": 180,
    "reddit-clustering": 50,
    "birds": 525,
}

alg_name_map = {"Vamana": "PECANN", "kmeans": "k-means"}


def plot_ari_by_cluster_offset_mult_figures(csv_path):
    plt.clf()
    set_superplot_font_sizes()

    df = pd.read_csv(csv_path)
    num_datasets = len(gt_num_clusters)
    num_cols = 5
    num_rows = (num_datasets + num_cols - 1) // num_cols
    plot_scaler = 6

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(plot_scaler * num_cols, plot_scaler * num_rows),
        tight_layout=True,
    )
    axes = axes.reshape(-1)

    dataset_groups = df.groupby("dataset")

    for (dataset_name, dataset_group), ax in zip(dataset_groups, axes):
        for method_name, method_group in dataset_group.groupby("method"):
            method_group["cluster_ratio"] = (
                method_group["num_clusters"] / gt_num_clusters[dataset_name]
            )
            method_group = method_group.sort_values("cluster_ratio")
            method_name = alg_name_map[method_name]
            ax.plot(
                method_group["cluster_ratio"],
                method_group["ARI"],
                label=f"{method_name}",
            )
            ax.axvline(1, c="black", linestyle=":")
            # ax.text(1.1, 0, "Correct number of clusters", rotation=90)

        dataset_name = dataset_name_map[dataset_name]
        ax.set_title(dataset_name)

    for i in range(num_datasets, num_rows * num_cols):
        axes[i].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, ncol=2, bbox_to_anchor=(0.25, 0.18))

    fig.supxlabel("Cluster Ratio: # Clusters Used / # Clusters in Ground Truth")
    fig.supylabel("ARI")
    plt.tight_layout()
    plt.savefig("results/paper/varying_num_clusters_all.pdf")

    reset_font_sizes()


def plot_ari_by_cluster_offset_one_figure_ours(csv_path):
    plt.clf()
    df = pd.read_csv(csv_path)

    grouped_data = df.groupby(["dataset", "method"])

    for (dataset, method), group in grouped_data:
        if method != "Vamana":
            continue
        group["cluster_ratio"] = group["num_clusters"] / gt_num_clusters[dataset]

        plt.plot(group["cluster_ratio"], group["ARI"], label=f"{dataset}")

    plt.title('Effect of Clustering with the "Wrong" Number of Clusters')
    plt.axvline(1, c="black", linestyle=":")
    plt.text(1.1, 0, "Correct number of clusters", rotation=90)
    plt.xlabel("Cluster Ratio: # Clusters Used / # Clusters in Ground Truth")
    plt.ylabel("ARI")
    plt.legend(title="Dataset")
    plt.savefig("results/paper/varying_num_clusters_vamana.pdf")


def pareto_front(x, y):
    sorted_indices = sorted(range(len(x)), key=lambda k: -x[k])
    x_sorted = [x[i] for i in sorted_indices]
    y_sorted = [y[i] for i in sorted_indices]

    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]

    for i in range(1, len(x_sorted)):
        if y_sorted[i] > pareto_front_y[-1]:
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])

    return pareto_front_x, pareto_front_y


import seaborn as sns


def plot_homogeneity_vs_completeness_pareto(csv_path):
    plt.clf()

    df = pd.read_csv(csv_path)

    grouped_data = df.groupby(["dataset", "method"])

    new_dataset = []

    for (dataset, method), group in grouped_data:
        group = group[group["homogeneity"] != 0]
        group = group[group["completeness"] != 0]
        xs, ys = group["homogeneity"].to_numpy(), group["completeness"].to_numpy()
        xs, ys = pareto_front(xs, ys)
        for x, y in zip(xs, ys):
            new_dataset.append((dataset, method, x, y))

    new_dataset = pd.DataFrame(new_dataset, columns=["dataset", "method", "x", "y"])
    sns.lineplot(
        new_dataset, x="x", y="y", hue="dataset", style="method", linewidth=2.5
    )

    plt.xlabel("Homogeneity")
    plt.ylabel("Completeness")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(
        "results/paper/varying_num_clusters_homogeneity_vs_completeness.pdf",
        bbox_inches="tight",
    )


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

    plot_homogeneity_vs_completeness_pareto(args.csv_path)
    plot_ari_by_cluster_offset_one_figure_ours(args.csv_path)
    plot_ari_by_cluster_offset_mult_figures(args.csv_path)

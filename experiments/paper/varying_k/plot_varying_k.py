import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from pathlib import Path
import sys
from adjustText import adjust_text

abspath = Path(__file__).resolve().parent.parent
sys.path.append(str(abspath))
from plotting_utils import set_superplot_font_sizes, reset_font_sizes, dataset_name_map


def plot_time_breakdown(df, dataset, ax):
    time_columns = [
        "Built index time",
        "Compute dependent points time",
        "Find knn time",
        "Compute density time",
        "Find clusters time",
    ]

    to_plot = df[time_columns]

    to_plot.plot.barh(stacked=True, ax=ax)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label_col"])

    ax.get_legend().remove()

    dataset = dataset_name_map[dataset]
    ax.set_title(dataset)


def plot_ari_vs_cluster_time(df, dataset, ax):
    x_col = "Total time"
    y_col = "ARI"

    density_groups = df.groupby("density_method")

    markers = ["<", "x", "v", "o", "P"]
    for (name, group), marker in zip(density_groups, markers):
        ax.scatter(group[x_col], group[y_col], label=name, marker=marker, s=150)

    texts = [
        ax.text(
            df[x_col][i],
            df[y_col][i],
            df["K"][i],
            ha="center",
            va="center",
            fontsize=18,
        )
        for i in range(len(df))
    ]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="black", alpha=0.8))

    dataset = dataset_name_map[dataset]
    ax.set_title(dataset)


def plot_combined_plots(folder_path):
    set_superplot_font_sizes()

    file_pattern = os.path.join(folder_path, "*_varying_k*.csv")
    csv_files = glob.glob(file_pattern)

    num_plots = len(csv_files)
    num_cols = 5
    num_rows = (num_plots + num_cols - 1) // num_cols
    plot_scaler = 6

    for title, plot_method in ["Clustering Time Breakdown", plot_time_breakdown], [
        "ARI vs Clustering Time",
        plot_ari_vs_cluster_time,
    ]:
        is_clustering_time = title == "Clustering Time Breakdown"
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(plot_scaler * num_cols, plot_scaler * num_rows),
            tight_layout=True,
            sharey=is_clustering_time,
        )
        axes = axes.reshape(-1)
        for i, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file)

            label_col = "label_col"
            df[label_col] = df["method"].str.split("_").str[-2:]
            df["K"] = df[label_col].str[1]
            df["density_method"] = df[label_col].str[0]
            dataset = df["dataset"][0]

            plot_method(df, dataset, axes[i])

        for i in range(num_plots, num_rows * num_cols):
            axes[i].axis("off")

        # plt.suptitle(title)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 0.8))

        combined_title = "_".join(title.split(" "))

        if not is_clustering_time:
            fig.supxlabel("Clustering Time (s)")
            fig.supylabel("ARI")
        else:
            fig.supxlabel("Time (s)")
            fig.supylabel("Density Method")

        plt.tight_layout()

        plt.savefig(f"results/paper/combined_{combined_title}.pdf", bbox_inches="tight")

    reset_font_sizes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot varying k and density for each dataset"
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folders containing the results of the varying k experiment",
    )

    args = parser.parse_args()

    plot_combined_plots(args.folder)

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

Path("results/graphs").mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", 500)

colors = {
    "Vamana": "tab:blue",
    "pyNNDescent": "tab:green",
    "HCNNG": "tab:orange",
    "kmeans": "tab:red",
    "fastdp": "tab:purple",
}
methods = ["Vamana", "pyNNDescent", "HCNNG", "kmeans", "fastdp"]


def pareto_front(x, y):
    sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
    x_sorted = [x[i] for i in sorted_indices]
    y_sorted = [y[i] for i in sorted_indices]

    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]

    for i in range(1, len(x_sorted)):
        if y_sorted[i] > pareto_front_y[-1]:
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])

    return pareto_front_x, pareto_front_y


def plot_pareto(ax, comparison, method, df):
    if len(df) == 0:
        return
    x, y = pareto_front(df["Total time"].to_numpy(), df["ARI"].to_numpy())

    display_method = method
    if method == "fastdp":
        display_method = "(scaled) fastdp"

    ax.plot(
        x,
        y,
        marker="o" if comparison == "ground truth" else "s",
        color=colors[method],
        linestyle="--" if comparison == "ground truth" else "-",
        label=f"{display_method} vs. {comparison}",
    )


def create_combined_pareto_plots(df):
    df.loc[df["method"].str.contains("fastdp"), "Total time"] /= 60

    # Because some floats are too long for pandas to do this normally?
    df["ARI"] = pd.to_numeric(df["ARI"])
    df["Total time"] = pd.to_numeric(df["Total time"])

    num_plots = df["dataset"].nunique()
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    plot_scaler = 6

    for comparison in ["ground truth", "brute force"]:
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(plot_scaler * num_cols, plot_scaler * num_rows),
            tight_layout=True,
        )
        filtered_df = df[df["comparison"] == comparison]
        dataset_groups = filtered_df.groupby("dataset")

        for i, (dataset_name, dataset_group) in enumerate(dataset_groups):
            current_axis = axes[i % num_rows][i // num_rows]
            for method in methods:
                more_filtered_df = dataset_group[
                    dataset_group["method"].str.contains(method)
                ]

                plot_pareto(current_axis, comparison, method, more_filtered_df)
                current_axis.set_title(dataset_name, fontsize=16)
                current_axis.set_xlabel("Clustering Time (s)", fontsize=16)
                current_axis.set_ylabel("ARI", fontsize=16)

        for i in range(num_plots, num_rows * num_cols):
            axes[i % num_rows][i // num_rows].axis("off")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.68, 0.3), fontsize=18)

        if comparison == "ground truth":
            plt.suptitle(
                "Pareto Front of ARI vs. Time, Comparing To Ground Truth", fontsize=20
            )
        else:
            plt.suptitle(
                "Pareto Front of ARI vs. Time, Comparing To Brute Force", fontsize=20
            )

        plt.tight_layout()

        plt.savefig(
            f"results/paper/pareto_frontier_plot_{comparison}.pdf",
            bbox_inches="tight",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot a pareto frontier of total time vs. AMI"
    )
    parser.add_argument("folder", type=str, help="Folder to read csv files from.")
    args = parser.parse_args()

    # Use glob to find files matching the pattern
    csv_files = glob.glob(args.folder + "/*pareto*.csv")
    df = pd.concat([pd.read_csv(path) for path in csv_files])
    create_combined_pareto_plots(df)


if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob


from pathlib import Path
import sys

abspath = Path(__file__).resolve().parent.parent
sys.path.append(str(abspath))
from plotting_utils import set_superplot_font_sizes, reset_font_sizes, dataset_name_map

Path("results/graphs").mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", 500)

colors = {
    "Vamana": "tab:blue",
    "pyNNDescent": "tab:green",
    "HCNNG": "tab:orange",
    "kmeans": "tab:red",
    "fastdp": "tab:purple",
    "DBSCAN": "tab:brown"
}


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


def plot_pareto(ax, comparison, method, df, map_method_name=True):
    if len(df) == 0:
        return
    x, y = pareto_front(df["Total time"].to_numpy(), df["ARI"].to_numpy())

    # print(f"{comparison},{method},{x[-1]},{y[-1]},{df['dataset'].iloc[0]}")
    display_method = method
    if method == "fastdp":
        display_method = "(scaled) fastdp"

    name_map = {"Vamana": "PECANN", "kmeans": "k-means"}
    if map_method_name:
        if display_method in name_map:
            display_method = name_map[display_method]

    ax.plot(
        x,
        y,
        marker="o",
        color=colors[method],
        linestyle="-",
        label=f"{display_method}",
    )


def create_combined_pareto_plots(df):
    plt.clf()

    set_superplot_font_sizes()

    methods = ["Vamana", "kmeans", "fastdp", "DBSCAN"]

    df.loc[df["method"].str.contains("fastdp"), "Total time"] /= 60

    # Because some floats are too long for pandas to do this normally?
    df["ARI"] = pd.to_numeric(df["ARI"])
    df["Total time"] = pd.to_numeric(df["Total time"])

    num_plots = df["dataset"].nunique()
    num_cols = 5
    num_rows = (num_plots + num_cols - 1) // num_cols
    plot_scaler = 6

    for comparison in ["ground truth", "brute force"]:
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(plot_scaler * num_cols, plot_scaler * num_rows),
        )
        axes = axes.reshape(-1)
        filtered_df = df[df["comparison"] == comparison]
        dataset_groups = filtered_df.groupby("dataset")

        for i, (dataset_name, dataset_group) in enumerate(dataset_groups):
            dataset_name = dataset_name_map[dataset_name]
            current_axis = axes[i]
            for method in methods:
                more_filtered_df = dataset_group[
                    dataset_group["method"].str.contains(method)
                ]

                plot_pareto(current_axis, comparison, method, more_filtered_df)
            current_axis.set_title(dataset_name)

        for i in range(num_plots, num_rows * num_cols):
            axes[i].axis("off")

        fig.supxlabel("Clustering Time (s)")
        supylabel = fig.supylabel("ARI")
        supylabel.set_position((0.005, 0.5))

        # if comparison == "ground truth":
        #     plt.suptitle("Pareto Front of ARI vs. Time, Comparing To Ground Truth")
        # else:
        #     plt.suptitle("Pareto Front of ARI vs. Time, Comparing To Brute Force")

        if comparison == "ground truth":
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.2, 0.1)
            )

        plt.tight_layout()

        plt.savefig(
            f"results/paper/pareto_frontier_plot_{comparison}.pdf",
            bbox_inches="tight",
        )
    reset_font_sizes()


def create_imagenet_different_graph_methods(df):
    plt.rcParams.update({"font.size": 22})
    methods = ["Vamana", "pyNNDescent", "HCNNG"]

    df = df[df["dataset"] == "imagenet"]
    for comparison in ["ground truth", "brute force"]:
        plt.clf()
        filtered_df = df[df["comparison"] == comparison]
        for method in methods:
            more_filtered_df = filtered_df[filtered_df["method"].str.contains(method)]

            plot_pareto(
                plt, comparison, method, more_filtered_df, map_method_name=False
            )

        if comparison == "ground truth":
            plt.legend()
        plt.xlabel("Clustering Time (s)")
        plt.ylabel("ARI")

        plt.savefig(
            f"results/paper/different_methods_on_imagenet_{comparison}.pdf",
            bbox_inches="tight",
        )


def create_table(df):
    df = df[df["comparison"] == "ground truth"]
    dataset_groups = df.groupby("dataset")
    methods = ["Vamana", "fastdp", "BruteForce", "kmeans"]

    for dataset_name, dataset_group in dataset_groups:
        dataset_name = dataset_name_map[dataset_name]
        print("\midrule")
        for method in methods:
            filtered_df = dataset_group[dataset_group["method"].str.contains(method)]

            x, y = pareto_front(
                filtered_df["Total time"].to_numpy(), filtered_df["ARI"].to_numpy()
            )
            for i in range(len(y)):
                if abs(y[i] - y[-1]) < 0.003:
                    break

            x, y = x[i], y[i]

            if method == "fastdp":
                x /= 60
                method = "fastdp (scaled)"

            if method == "Vamana":
                method = "\\framework"
            else:
                method = f"\\algname{{{method}}}"

            print(f"{method} & \\datasetname{{{dataset_name}}} & {x:.2f} & {y:.2f}\\\\")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a pareto frontier of total time vs. ARI"
    )
    parser.add_argument("folder", type=str, help="Folder to read csv files from.")
    args = parser.parse_args()

    # Use glob to find files matching the pattern
    csv_files = glob.glob(args.folder + "/*pareto*.csv")
    brute_force_files = glob.glob(args.folder + "/*brute*.csv")
    csv_files += brute_force_files
    df = pd.concat([pd.read_csv(path) for path in csv_files])

    ## Read DBSCAN files
    dbscan_files = glob.glob(args.folder + "/*dbscan*.csv")
    df2 = pd.concat([pd.read_csv(path) for path in dbscan_files])
    df2["method"] = "DBSCAN"
    df2["num_threads"] = 30
    df2["comparison"] = "ground truth"
    df2.rename(columns={"sklearn_time": "Total time"}, inplace=True)

    df = pd.concat([df, df2])

    create_table(df)
    create_imagenet_different_graph_methods(df)
    create_combined_pareto_plots(df)


if __name__ == "__main__":
    main()

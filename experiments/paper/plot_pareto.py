import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

Path("results/graphs").mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", 500)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a pareto frontier of total time vs. AMI"
    )
    parser.add_argument("folder", type=str, help="Folder to read csv files from.")
    args = parser.parse_args()

    # Use glob to find files matching the pattern
    csv_files = glob.glob(args.folder + "/*pareto*.csv")
    df = pd.concat([pd.read_csv(path) for path in csv_files])

    # Because some floats are too long for pandas to do this normally?
    df["ARI"] = pd.to_numeric(df["ARI"])
    df["Total time"] = pd.to_numeric(df["Total time"])

    colors = {
        "Vamana": "tab:blue",
        "pyNNDescent": "tab:green",
        "HCNNG": "tab:orange",
        "kmeans": "tab:red",
        "fastdp": "tab:purple",
    }
    methods = ["Vamana", "pyNNDescent", "HCNNG", "kmeans", "fastdp"]

    dfs = {}

    for comparison in ["ground truth", "brute force"]:
        for method in methods:
            filtered_df = df[df["comparison"] == comparison]
            filtered_df = filtered_df[filtered_df["method"].str.contains(method)]

            filtered_df = filtered_df.sort_values(by=["Total time"])
            pareto_frontier = []
            max_ari = float("-inf")

            for _, row in filtered_df.iterrows():
                ari = row["ARI"]
                if ari > max_ari:
                    max_ari = ari
                    pareto_frontier.append(row)

            pareto_df = pd.DataFrame(pareto_frontier)

            dfs[(comparison, method)] = (filtered_df, pareto_df)

    def plot_pareto(comparison, method):
        _, pareto_df = dfs[((comparison, method))]
        if len(pareto_df) == 0:
            return
        plt.plot(
            pareto_df["Total time"],
            pareto_df["ARI"],
            marker="o" if comparison == "ground truth" else "s",
            color=colors[method],
            linestyle="--" if comparison == "ground truth" else "-",
            label=f"{method} vs. {comparison}",
        )

    def new_plot(title):
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.xlabel("Total clustering time (s)")
        plt.ylabel("ARI")
        plt.title(title)
        plt.grid(True)

    for comparison in ["ground truth", "brute force"]:
        new_plot(title=f"ARI vs. clustering time")
        for method in methods:
            plot_pareto(comparison, method)

        plt.legend(loc="lower right")
        plt.savefig(
            f"results/paper/pareto_frontier_plot_{comparison}.pdf",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()

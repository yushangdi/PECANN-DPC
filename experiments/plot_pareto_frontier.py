import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("results/graphs").mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", 500)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a pareto frontier of total time vs. AMI"
    )
    parser.add_argument("file_path", type=str, help="Path to the CSV results file")
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

    colors = {"Vamana": "tab:blue", "pyNNDescent": "tab:green", "HCNNG": "tab:orange"}

    dfs = {}

    for comparison in ["ground truth", "brute force"]:
        for graph_type in ["Vamana", "pyNNDescent", "HCNNG"]:
            filtered_df = df[df["comparison"] == comparison]
            filtered_df = filtered_df[filtered_df["method"].str.contains(graph_type)]

            filtered_df = filtered_df.sort_values(by=["Total time"])

            pareto_frontier = []
            max_ari = float("-inf")

            for _, row in filtered_df.iterrows():
                ari = row["ARI"]
                if ari > max_ari:
                    max_ari = ari
                    pareto_frontier.append(row)

            pareto_df = pd.DataFrame(pareto_frontier)

            dfs[(comparison, graph_type)] = (filtered_df, pareto_df)

    def plot_pareto(comparison, graph_type):
        _, pareto_df = dfs[((comparison, graph_type))]
        if len(pareto_df) == 0:
            return
        plt.plot(
            pareto_df["Total time"],
            pareto_df["ARI"],
            marker="o" if comparison == "ground truth" else "s",
            color=colors[graph_type],
            linestyle="--" if comparison == "ground truth" else "-",
            label=f"{graph_type} vs. {comparison}",
        )

    def new_plot(title):
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.xlabel("Total clustering time (s)")
        plt.ylabel("ARI")
        plt.title(title)
        plt.grid(True)

    new_plot(title=f'{df.iloc[0]["dataset"]} ARI vs. clustering time pareto frontiers')
    for comparison in ["ground truth", "brute force"]:
        for graph_type in ["Vamana", "pyNNDescent", "HCNNG"]:
            plot_pareto(comparison, graph_type)

    plt.legend(loc="lower right")
    plt.savefig("results/graphs/pareto_frontier_plot.png", bbox_inches="tight")

    for graph_type in ["Vamana", "pyNNDescent", "HCNNG"]:
        new_plot(title=f'{graph_type} grid search on {df.iloc[0]["dataset"]}')
        comparison = "ground truth"
        plot_pareto(comparison, graph_type)
        filtered_df, pareto_df = dfs[((comparison, graph_type))]
        plt.scatter(
            filtered_df["Total time"],
            filtered_df["ARI"],
            color=colors[graph_type],
            alpha=0.5,
        )

        plt.legend(loc="lower right")
        plt.savefig(f"results/graphs/{graph_type}_vs_gt.png", bbox_inches="tight")


if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a pareto frontier of total time vs. AMI"
    )
    parser.add_argument("file_path", type=str, help="Path to the CSV results file")
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

    plt.figure(figsize=(10, 6))

    # Create a plot to combine all Pareto frontiers
    plt.figure(figsize=(10, 6))
    plt.xlabel("Total clustering time (s)")
    plt.ylabel("ARI")
    plt.title(f'{df.iloc[0]["dataset"]} ARI vs. Clustering time')

    for comparison in ["ground truth", "brute force"]:
        filtered_df = df[df["comparison"] == comparison]

        filtered_df = filtered_df.sort_values(by=["Total time"])

        pareto_frontier = []
        max_ari = float("-inf")

        # Iterate through the sorted DataFrame to find the Pareto frontier
        for _, row in filtered_df.iterrows():
            ari = row["ARI"]
            if ari > max_ari:
                max_ari = ari
                pareto_frontier.append(row)

        # Convert the Pareto frontier to a DataFrame
        pareto_df = pd.DataFrame(pareto_frontier)
        # print(pareto_df[["method", "Total time", "ARI"]])
        plt.plot(
            pareto_df["Total time"],
            pareto_df["ARI"],
            marker="o",
            linestyle="-",
            label=f"Comparing against {comparison}",
        )

    plt.legend()
    plt.grid(True)
    plt.savefig("pareto_frontier_plot.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

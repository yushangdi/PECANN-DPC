import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import ListedColormap

def plot_time_breakdown(df, dataset):

    time_columns = ["Built index time", "Compute dependent points time", "Find knn time", "Compute density time", "Find clusters time"]

    plt.figure()
    for i, (_, row) in enumerate(df.iterrows()):
        plt.barh(i, row[time_columns], color=plt.cm.Paired.colors)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Density Method')
    plt.legend(time_columns, loc='upper right', bbox_to_anchor=(1.25, 1))
    for i in range(len(time_columns)):
        plt.gca().get_legend().legendHandles[i].set_color(plt.cm.Paired.colors[i])

    plt.yticks(range(len(df)), df['label_col'])

    # Add title
    plt.title('Clustering Time Breakdown')

    # Save the plot
    plt.savefig(f'results/paper/time_breakdown_{dataset}.png')


def plot_ari_vs_cluster_time(df, dataset):

    x_col = "Total time"
    y_col = "ARI"

    plt.figure()
    density_groups = df.groupby("density_method")

    for name, group in density_groups:
        plt.scatter(group[x_col], group[y_col], label=name)

    for i, label in enumerate(df["K"]):
        plt.annotate(
            label,
            (df[x_col][i], df[y_col][i]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="right",
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(loc='lower right')
    plt.title(f"ARI vs Clustering Time For {dataset}")

    plt.savefig(f"results/paper/varying_k_{dataset}.png")
    
def plot_varying_k(folder_path):
    file_pattern = os.path.join(folder_path, "*_varying_k*.csv")
    csv_files = glob.glob(file_pattern)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        label_col = "label_col"
        df[label_col] = df["method"].str.split("_").str[-2:]
        df["K"] = df[label_col].str[1]
        df["density_method"] = df[label_col].str[0]
        dataset = df["dataset"][0]

        plot_ari_vs_cluster_time(df, dataset)
        plot_time_breakdown(df, dataset)
        


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

    plot_varying_k(args.folder)

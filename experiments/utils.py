import time

# Should already have parent folder on path for this to work
from post_processors.cluster_eval import eval_cluster_files
import os
import pandas as pd
import sys
from union_find import UnionFind
import numpy as np

quality_headers = [
    "recall50",
    "precision50",
    # "AMI",
    "ARI",
    "completeness",
    "homogeneity",
]
time_check_headers = [
    "Built index",
    "Compute dependent points",
    "Compute density",
    "Find clusters",
    "Total",
]
headers = [t + " time" for t in time_check_headers] + quality_headers


def _cluster_by_densities_distance_product(
    graph, num_clusters, density_product, distance_product
):
    parents = graph["Parent_ID"].to_numpy(copy=True)
    new_column = (np.log(graph["Density"]) * density_product) + (
        np.log(graph["Parent_Distance"]) * distance_product
    )
    top_k_densities = new_column.nlargest(num_clusters)
    parents[top_k_densities.index] = -1

    u = UnionFind()
    for i, p in enumerate(parents):
        if p != -1:
            u.unite(i, p)
        else:
            u.add(i)

    return np.array([u.find(i) for i in range(len(graph))])


def product_cluster_dg(dg_path, num_clusters, density_product=1, distance_product=1):
    graph = pd.read_csv(
        dg_path, delimiter=" ", names=["Density", "Parent_Distance", "Parent_ID"]
    )

    # Force parentless nodes to be selected by any strategy
    parentless = graph["Parent_Distance"] == -1
    graph.loc[parentless, "Density"] = 1
    graph.loc[parentless, "Parent_Distance"] = float("inf")

    # Force duplicates to be never selected by any strategy
    duplicates = graph["Parent_Distance"] == 0
    graph.loc[duplicates, "Parent_Distance"] = sys.float_info.min

    return _cluster_by_densities_distance_product(
        graph,
        num_clusters=num_clusters,
        density_product=density_product,
        distance_product=distance_product,
    )


def create_results_file():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cluster_results_file = f"results/cluster_analysis_{timestr}.csv"

    with open(cluster_results_file, "w") as f:
        f.write("dataset,method,comparison," + ",".join(headers) + "\n")

    return cluster_results_file


def eval_cluster_and_write_results(
    gt_cluster_path,
    cluster_path,
    compare_to_ground_truth,
    results_file,
    dataset,
    graph_type,
    time_reports,
):
    times = [
        (str(time_reports[key]) if key in time_reports else "")
        for key in time_check_headers
    ]
    cluster_results = eval_cluster_files(
        gt_path=gt_cluster_path,
        cluster_path=cluster_path,
        verbose=False,
        metrics=quality_headers,
    )
    with open(results_file, "a") as f:
        fields = (
            [
                dataset,
                graph_type,
                "ground truth" if compare_to_ground_truth else "brute force",
            ]
            + times
            + [str(cluster_results[h]) for h in quality_headers]
        )
        f.write(",".join(fields) + "\n")


def make_results_folder(dataset):
    if dataset == "s2" or dataset == "s3":
        dataset_folder = "s_datasets"
    else:
        dataset_folder = dataset
    os.makedirs(f"results/{dataset_folder}", exist_ok=True)
    return dataset_folder


def get_cutoff(dataset):
    # From analyzing decision graph
    return {
        "mnist": {"distance_cutoff": 3, "center_density_cutoff": 0.7},
        "s2": {"distance_cutoff": 102873},
        "s3": {"distance_cutoff": 102873},
        "unbalance": {"distance_cutoff": 30000},
    }[dataset]

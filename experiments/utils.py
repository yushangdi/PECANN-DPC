import time

# Should already have parent folder on path for this to work
from post_processors.cluster_eval import eval_clusters_wrapper
import os
import pandas as pd
import sys
from union_find import UnionFind
import numpy as np
import dpc_ann

quality_headers = [
    "recall50",
    "precision50",
    # "AMI",
    "ARI",
    "completeness",
    "homogeneity",
]
time_check_headers = [
    "Built index time",
    "Compute dependent points time",
    "Find knn time",
    "Compute density time",
    "Find clusters time",
    "Total time",
]
misc_headers = ["dataset", "method", "comparison", "num_threads"]
headers = misc_headers + time_check_headers + quality_headers


def _num_threads():
    return len(os.sched_getaffinity(0))


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


def create_results_file(prefix=""):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cluster_results_file = f"results/cluster_analysis_{prefix}_{timestr}.csv"

    with open(cluster_results_file, "w") as f:
        f.write(",".join(headers) + "\n")

    return cluster_results_file


def eval_cluster_and_write_results(
    gt_cluster_path,
    found_clusters,
    compare_to_ground_truth,
    results_file,
    dataset,
    method,
    time_reports,
):
    # TODO(Josh): Can clean this up a bit when deleting the old DPC code
    adjusted_time_reports = {
        a + (" time" if not a.endswith(" time") else ""): b
        for a, b in time_reports.items()
    }
    times = [
        (str(adjusted_time_reports[key]) if key in adjusted_time_reports else "")
        for key in time_check_headers
    ]
    cluster_results = eval_clusters_wrapper(
        gt_path=gt_cluster_path,
        found_clusters=found_clusters,
        verbose=False,
        eval_metrics=quality_headers,
    )
    with open(results_file, "a") as f:
        fields = (
            [
                dataset,
                method,
                "ground truth" if compare_to_ground_truth else "brute force",
                str(_num_threads()),
            ]
            + times
            + [str(cluster_results[h]) for h in quality_headers]
        )
        f.write(",".join(fields) + "\n")
    return cluster_results


def make_results_folder(dataset):
    if dataset == "s2" or dataset == "s3":
        dataset_folder = "s_datasets"
    else:
        dataset_folder = dataset
    os.makedirs(f"results/{dataset_folder}", exist_ok=True)
    return dataset_folder


def get_threshold_center_finder(dataset):
    # From analyzing decision graph
    settings = {
        "mnist": {"dependant_dist_threshold": 3, "density_threshold": 0.7},
        "s2": {"dependant_dist_threshold": 102873},
        "s3": {"dependant_dist_threshold": 102873},
        "unbalance": {"dependant_dist_threshold": 30000},
    }[dataset]
    return {"center_finder": dpc_ann.ThresholdCenterFinder(**settings)}

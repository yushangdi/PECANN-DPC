import time

# Should already have parent folder on path for this to work
from post_processors.cluster_eval import eval_cluster
import os

quality_headers = [
    "recall50",
    "precision50",
    "AMI",
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


def get_times_from_stdout(keys, stdout):
    result = [""] * len(keys)
    split_lines = [line.split(":") for line in str(stdout).split("\n")]
    split_lines = [line for line in split_lines if len(line) == 2]
    for header, value in split_lines:
        if header in keys:
            result[keys.index(header)] = value.strip()
    return result


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
    method,
    dpc_stdout,
):
    times = get_times_from_stdout(keys=time_check_headers, stdout=dpc_stdout)
    cluster_results = eval_cluster(
        gt_path=gt_cluster_path, cluster_path=cluster_path, verbose=False
    )
    with open(results_file, "a") as f:
        fields = (
            [
                dataset,
                method,
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
        "unbalance": {"distance_cutoff": 3},
    }[dataset]

#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path

from data_processors.plot import plot_dims
from post_processors.cluster_eval import eval_cluster

# Change to DPC-ANN folder
abspath = Path(__file__).resolve().parent
os.chdir(abspath)

cluster_results_file = "results/cluster_analysis.csv"
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
with open(cluster_results_file, "w") as f:
    f.write("dataset,method,comparison," + ",".join(headers) + "\n")

# Used to determine cluster centroids (from analyzing decision graph)
cutoffs = {
    "mnist": "--dist_cutoff 3 --center_density_cutoff 0.7 ",
    "s2": "--dist_cutoff 102873 ",
    "s3": "--dist_cutoff 102873 ",
    "unbalance": "--dist_cutoff 30000 ",
}


def get_times_from_stdout(keys, stdout):
    stdout = stdout.decode("utf-8")
    result = [""] * len(keys)
    split_lines = [line.split(":") for line in str(stdout).split("\n")]
    split_lines = [line for line in split_lines if len(line) == 2]
    for header, value in split_lines:
        if header in keys:
            result[keys.index(header)] = value.strip()
    return result


for dataset in ["s2", "mnist", "s3", "unbalance"]:
    for method in ["bruteforce", "HCNNG", "pyNNDescent", "Vamana"]:
        if dataset == "s2" or dataset == "s3":
            dataset_folder = "s_datasets"
        else:
            dataset_folder = dataset

        query_file = f"data/{dataset_folder}/{dataset}.txt"

        os.makedirs(f"results/{dataset_folder}", exist_ok=True)
        prefix = f"results/{dataset_folder}/{dataset}_{method}"
        dpc_command = (
            f"./doubling_dpc --query_file {query_file} "
            + f"--decision_graph_path {prefix}.dg "
            + f"--output_file {prefix}.cluster "
            + cutoffs[dataset]
        )
        if method == "bruteforce":
            dpc_command += f"--bruteforce true "
        else:
            dpc_command += f"--graph_type {method}"

        # Run DPC
        stdout = subprocess.check_output(dpc_command, shell=True)
        times = get_times_from_stdout(keys=time_check_headers, stdout=stdout)

        # Eval cluster against ground truth and write results
        cluster_results = eval_cluster(
            gt_path=f"data/{dataset_folder}/{dataset}.gt",
            cluster_path=f"{prefix}.cluster",
        )
        with open(cluster_results_file, "a") as f:
            fields = (
                [dataset, method, "ground truth"]
                + times
                + [str(cluster_results[h]) for h in quality_headers]
            )
            f.write(",".join(fields) + "\n")

        # Eval cluster against brute force
        if method != "bruteforce":
            cluster_results = eval_cluster(
                gt_path=f"results/{dataset_folder}/{dataset}_bruteforce.cluster",
                cluster_path=f"{prefix}.cluster",
            )
            with open(cluster_results_file, "a") as f:
                fields = (
                    [dataset, method, "bruteforce"]
                    + times
                    + [str(cluster_results[h]) for h in quality_headers]
                )
                f.write(",".join(fields) + "\n")

        # Plot clusters
        plot_dims(
            filename=query_file,
            cluster_path=f"{prefix}.cluster",
            image_path=f"{prefix}.png",
        )

#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path
import sys

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from data_processors.plot import plot_dims

from utils import (
    create_results_file,
    eval_cluster_and_write_results,
    make_results_folder,
)


cluster_results_file = create_results_file()

# Used to determine cluster centroids (from analyzing decision graph)
cutoffs = {
    "mnist": "--dist_cutoff 3 --center_density_cutoff 0.7 ",
    "s2": "--dist_cutoff 102873 ",
    "s3": "--dist_cutoff 102873 ",
    "unbalance": "--dist_cutoff 30000 ",
}

for dataset in ["s2", "mnist", "s3", "unbalance"]:
    dataset_folder = make_results_folder(dataset)
    for method in ["bruteforce", "HCNNG", "pyNNDescent", "Vamana"]:
        query_file = f"data/{dataset_folder}/{dataset}.txt"
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

        # Eval cluster against ground truth and write results
        eval_cluster_and_write_results(
            gt_cluster_path=f"data/{dataset_folder}/{dataset}.gt",
            cluster_path=f"{prefix}.cluster",
            compare_to_ground_truth=True,
            results_file=cluster_results_file,
            dataset=dataset,
            method=method,
            dpc_stdout=stdout,
        )

        # Eval cluster against brute force DPC
        eval_cluster_and_write_results(
            gt_cluster_path=f"results/{dataset_folder}/{dataset}_bruteforce.cluster",
            cluster_path=f"{prefix}.cluster",
            compare_to_ground_truth=False,
            results_file=cluster_results_file,
            dataset=dataset,
            method=method,
            dpc_stdout=stdout,
        )

        # Plot clusters
        plot_dims(
            filename=query_file,
            cluster_path=f"{prefix}.cluster",
            image_path=f"{prefix}.png",
        )

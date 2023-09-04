#!/usr/bin/env python3

import itertools
import os
from pathlib import Path
import sys
import subprocess
from tqdm import tqdm

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from utils import (
    create_results_file,
    eval_cluster_and_write_results,
    make_results_folder,
    get_cutoff,
)

dataset = "mnist"

cluster_results_file = create_results_file()

options = []

# For now K and cutoffs are fixed since we are mostly comparing to brute force
# Also for now we're just going to look at Vamana to get decent parameters,
# can add others once we know reasonable parameters for num_clusters
#  and reduce the search space with the vamana search


exponential_range = [2, 4, 8, 16, 32, 64, 128]
for (
    max_degree,
    beam_search_construction,
    beam_search_clustering,
) in itertools.product(exponential_range, exponential_range, exponential_range):
    for alpha in [1, 1.1, 1.2, 1.3, 1.5, 2]:
        for beam_search_density in [8, 16, 32, 64, 128]:
            options.append(
                (
                    f"Vamana_{max_degree}_{alpha}_{beam_search_construction}_{beam_search_density}_{beam_search_clustering}",
                    f"--max_degree {max_degree} --alpha {alpha} --Lbuild {beam_search_construction} --L {beam_search_density} --Lnn {beam_search_clustering} ",
                )
            )

dataset_folder = make_results_folder(dataset)

for method, command in tqdm(options):
    query_file = f"data/{dataset_folder}/{dataset}.txt"
    prefix = f"results/{dataset_folder}/{dataset}_{method}"

    dpc_command = (
        f"./doubling_dpc --query_file {query_file} "
        + f"--decision_graph_path {prefix}.dg "
        + f"--output_file {prefix}.cluster "
        + get_cutoff(dataset)
        + command
    )

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
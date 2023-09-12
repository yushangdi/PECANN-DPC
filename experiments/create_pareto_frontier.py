#!/usr/bin/env python3

import itertools
import os
from pathlib import Path
import sys
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

import dpc_ann

dataset = "mnist"

cluster_results_file = create_results_file()

options = []

# For now K and cutoffs are fixed since we are mostly comparing to brute force
# Also for now we're just going to look at Vamana to get decent parameters,
# can add others once we know reasonable parameters for num_clusters
#  and reduce the search space with the vamana search

# For now not including 128 at top of range

exponential_range = [4, 8, 16, 32, 64]
for (
    max_degree,
    beam_search_construction,
    beam_search_clustering,
) in itertools.product(exponential_range, exponential_range, exponential_range):
    # We are assuming Vamana value of alpha = 1.1 (experimentally verified) works well for other graph methods
    # TODO(Josh): Validate this assumption? Can just leave running in background somewhere
    # for alpha in [1, 1.05, 1.1, 1.15, 1.2]:
    for alpha in [1.1]:
        for beam_search_density in [8, 16, 32, 64]:
            for graph_type in ["Vamana", "pyNNDescent", "HCNNG"]:
                method = f"{graph_type}_{max_degree}_{alpha}_{beam_search_construction}_{beam_search_density}_{beam_search_clustering}"
                command_line = {
                    "max_degree": max_degree,
                    "alpha": alpha,
                    "Lbuild": beam_search_construction,
                    "L": beam_search_density,
                    "Lnn": beam_search_clustering,
                    "graph_type": graph_type,
                }
                if graph_type in ["pyNNDescent", "HCNNG"]:
                    if beam_search_construction < 8:
                        continue
                    for num_clusters in range(1, 6):
                        new_command_line = dict(command_line)
                        command_line["num_clusters"] = num_clusters
                        options.append((method, new_command_line))
                else:
                    options.append((method, command_line))

dataset_folder = make_results_folder(dataset)

for method, command in tqdm(options):
    query_file = f"data/{dataset_folder}/{dataset}.txt"
    prefix = f"results/{dataset_folder}/{dataset}_{method}"

    times = dpc_ann.dpc(
        **command,
        **get_cutoff(dataset),
        data_path=query_file,
        decision_graph_path=f"{prefix}.dg ",
        output_path=f"{prefix}.cluster",
    )

    # Eval cluster against ground truth and write results
    eval_cluster_and_write_results(
        gt_cluster_path=f"data/{dataset_folder}/{dataset}.gt",
        cluster_path=f"{prefix}.cluster",
        compare_to_ground_truth=True,
        results_file=cluster_results_file,
        dataset=dataset,
        graph_type=graph_type,
        time_reports=times,
    )

    # Eval cluster against brute force DPC
    eval_cluster_and_write_results(
        gt_cluster_path=f"results/{dataset_folder}/{dataset}_bruteforce.cluster",
        cluster_path=f"{prefix}.cluster",
        compare_to_ground_truth=False,
        results_file=cluster_results_file,
        dataset=dataset,
        graph_type=graph_type,
        time_reports=times,
    )

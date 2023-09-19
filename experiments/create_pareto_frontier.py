#!/usr/bin/env python3

import itertools
import os
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import multiprocessing
import argparse

import dpc_ann

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from utils import (
    create_results_file,
    eval_cluster_and_write_results,
    make_results_folder,
    product_cluster_dg,
)


parser = argparse.ArgumentParser(description="Process ground truth and dataset paths.")
parser.add_argument(
    "dataset",
    help="Dataset name (should be a file named data/<dataset>/<dataset>.npy and data/dataset/<dataset>gt).",
)
parser.add_argument(
    "num_clusters",
    help="How many clusters to use when generating the clusters from the decision graph files",
    type=int,
)
parser.add_argument(
    "--timeout",
    help="How long to wait in seconds before killing one of the running jobs",
    default=20,
    type=int,
)
args = parser.parse_args()

dataset = args.dataset
timeout_s = args.timeout
num_clusters = args.num_clusters


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
            # for graph_type in ["Vamana", "pyNNDescent", "HCNNG"]:
            for graph_type in ["Vamana"]:
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
                    if beam_search_construction < 16:
                        continue
                    for num_clusters in range(1, 6):
                        new_command_line = dict(command_line)
                        command_line["num_clusters"] = num_clusters
                        options.append((method, new_command_line))
                else:
                    options.append((method, command_line))

dataset_folder = make_results_folder(dataset)

data = np.load(f"data/{dataset_folder}/{dataset}.npy").astype("float32")


def create_product_clustering(decision_graph_path, num_clusters, output_path):
    clusters = product_cluster_dg(decision_graph_path, num_clusters)
    clusters = clusters.reshape((len(clusters), 1))
    np.savetxt(output_path, clusters, fmt="%i")


ground_truth_cluster_path = f"results/{dataset_folder}/{dataset}_BruteForce.cluster"
if not os.path.isfile(ground_truth_cluster_path):
    decision_graph_path = f"results/{dataset_folder}/{dataset}_BruteForce.dg"
    dpc_ann.dpc_numpy(
        graph_type="BruteForce",
        decision_graph_path=decision_graph_path,
        # output_path=ground_truth_cluster_path,
        # **get_cutoff(dataset),
        data=data,
    )
# Always recreate ground truth clustering
create_product_clustering(
    decision_graph_path, num_clusters, ground_truth_cluster_path
)


def try_command(graph_type, command):
    prefix = f"results/{dataset_folder}/{dataset}_{graph_type}"

    times = dpc_ann.dpc_numpy(
        **command,
        # **get_cutoff(dataset),
        data=data,
        decision_graph_path=f"{prefix}.dg",
        # output_path=f"{prefix}.cluster",
    )
    create_product_clustering(
        decision_graph_path=f"{prefix}.dg",
        num_clusters=num_clusters,
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
        gt_cluster_path=f"results/{dataset_folder}/{dataset}_BruteForce.cluster",
        cluster_path=f"{prefix}.cluster",
        compare_to_ground_truth=False,
        results_file=cluster_results_file,
        dataset=dataset,
        graph_type=graph_type,
        time_reports=times,
    )


for graph_type, command in tqdm(options):
    p = multiprocessing.Process(target=try_command, args=(graph_type, command))
    p.start()

    exitcode = p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        print(graph_type, "timed out!")
    elif exitcode != 0:
        print(graph_type, "had exit code", str(p.exitcode) + "!")

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
)


def run_dpc_ann_configurations(
    dataset,
    timeout_s,
    num_clusters,
    graph_types=None,
    search_range=None,
    compare_against_gt=True,
    results_file_prefix="",
):
    cluster_results_file = create_results_file(prefix=results_file_prefix)

    options = []

    if search_range == None:
        search_range = [8, 16, 32, 64]
        if dataset == "imagenet":
            search_range += [128, 256]

    if graph_types == None:
        graph_types = ["Vamana", "HCNNG", "pyNNDescent"]

    for (
        max_degree,
        beam_search_construction,
        beam_search_clustering,
        beam_search_density,
    ) in itertools.product(search_range, search_range, search_range, search_range):
        # We are assuming Vamana value of alpha = 1.1 (experimentally verified) works well for other graph methods
        # TODO(Josh): Validate this assumption? Can just leave running in background somewhere
        # for alpha in [1, 1.05, 1.1, 1.15, 1.2]:
        for alpha in [1.1]:
            for graph_type in graph_types:
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
                    for num_clusters in range(1, 4):
                        new_command_line = dict(command_line)
                        command_line["num_clusters"] = num_clusters
                        new_method = method + "_" + str(num_clusters)
                        options.append((new_method, new_command_line))
                else:
                    options.append((method, command_line))

    dataset_folder = make_results_folder(dataset)

    data = np.load(f"data/{dataset_folder}/{dataset}.npy").astype("float32")

    ground_truth_cluster_path = f"results/{dataset_folder}/{dataset}_BruteForce.cluster"
    ground_truth_decision_graph_path = (
        f"results/{dataset_folder}/{dataset}_BruteForce.dg"
    )
    if not os.path.isfile(ground_truth_cluster_path):
        dpc_ann.dpc_numpy(
            graph_type="BruteForce",
            decision_graph_path=ground_truth_decision_graph_path,
            output_path=ground_truth_cluster_path,
            data=data,
            center_finder=dpc_ann.ProductCenterFinder(num_clusters=num_clusters),
        )

    def try_command(graph_type, command):
        prefix = f"results/{dataset_folder}/{dataset}_{graph_type}_new"

        clustering_result = dpc_ann.dpc_numpy(
            **command,
            data=data,
            decision_graph_path=f"{prefix}.dg",
            center_finder=dpc_ann.ProductCenterFinder(num_clusters=num_clusters),
        )

        # Eval cluster against ground truth and write results
        if compare_against_gt:
            eval_cluster_and_write_results(
                gt_cluster_path=f"data/{dataset_folder}/{dataset}.gt",
                found_clusters=np.array(clustering_result.clusters),
                compare_to_ground_truth=True,
                results_file=cluster_results_file,
                dataset=dataset,
                method=graph_type,
                time_reports=clustering_result.metadata,
            )

        # Eval cluster against brute force DPC
        eval_cluster_and_write_results(
            gt_cluster_path=f"results/{dataset_folder}/{dataset}_BruteForce.cluster",
            found_clusters=np.array(clustering_result.clusters),
            compare_to_ground_truth=False,
            results_file=cluster_results_file,
            dataset=dataset,
            method=graph_type,
            time_reports=clustering_result.metadata,
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

    return cluster_results_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run approximate DPC methods on the passed in files."
    )
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
    parser.add_argument(
        "--results_file_prefix", help="Prefix for the results files", default=""
    )
    parser.add_argument("--dont_compare_against_gt", default=False, action="store_true")
    parser.add_argument("-search_range", nargs="+", type=int)
    parser.add_argument("-graph_types", nargs="+", type=str)
    args = parser.parse_args()

    run_dpc_ann_configurations(
        dataset=args.dataset,
        timeout_s=args.timeout,
        num_clusters=args.num_clusters,
        compare_against_gt=not args.dont_compare_against_gt,
        search_range=args.search_range,
        graph_types=args.graph_types,
        results_file_prefix=args.results_file_prefix,
    )

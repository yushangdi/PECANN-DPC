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


def run_dpc_ann_configurations(
    dataset,
    timeout_s,
    num_clusters,
    graph_types=None,
    search_range=None,
    compare_against_gt=True,
    run_new_dpc_framework=True,
    run_old_dpc_framework=False,
):
    if not run_new_dpc_framework and not run_old_dpc_framework:
        raise ValueError(
            "At least one of run_new_dpc_framework or run_old_dpc_framework must be true"
        )

    cluster_results_file = create_results_file()

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
                    for num_clusters in range(1, 6):
                        new_command_line = dict(command_line)
                        command_line["num_clusters"] = num_clusters
                        new_method = method + "_" + str(num_clusters)
                        options.append((new_method, new_command_line))
                else:
                    options.append((method, command_line))

    dataset_folder = make_results_folder(dataset)

    data = np.load(f"data/{dataset_folder}/{dataset}.npy").astype("float32")

    def create_product_clustering(decision_graph_path, num_clusters, output_path):
        clusters = product_cluster_dg(decision_graph_path, num_clusters)
        clusters = clusters.reshape((len(clusters), 1))
        np.savetxt(output_path, clusters, fmt="%i")

    ground_truth_cluster_path = f"results/{dataset_folder}/{dataset}_BruteForce.cluster"
    ground_truth_decision_graph_path = (
        f"results/{dataset_folder}/{dataset}_BruteForce.dg"
    )
    if not os.path.isfile(ground_truth_cluster_path):
        dpc_ann.dpc_numpy(
            graph_type="BruteForce",
            decision_graph_path=ground_truth_decision_graph_path,
            # output_path=ground_truth_cluster_path,
            # **get_cutoff(dataset),
            data=data,
        )
    # Always recreate ground truth clustering
    create_product_clustering(
        ground_truth_decision_graph_path, num_clusters, ground_truth_cluster_path
    )

    def try_command(graph_type, command, use_new_framework):
        prefix = f"results/{dataset_folder}/{dataset}_{graph_type}"
        if use_new_framework:
            prefix += "_new"

        clustering_result = dpc_ann.dpc_numpy(
            **command,
            # **get_cutoff(dataset),
            data=data,
            decision_graph_path=f"{prefix}.dg",
            use_new_framework=use_new_framework
            # output_path=f"{prefix}.cluster",
        )
        times = clustering_result.metadata

        # Create product clustering manually using dg file output
        # TODO: Add product clustering option to framework directly
        create_product_clustering(
            decision_graph_path=f"{prefix}.dg",
            num_clusters=num_clusters,
            output_path=f"{prefix}.cluster",
        )

        if use_new_framework:
            graph_type += "_new"

        # Eval cluster against ground truth and write results
        if compare_against_gt:
            eval_cluster_and_write_results(
                gt_cluster_path=f"data/{dataset_folder}/{dataset}.gt",
                cluster_path=f"{prefix}.cluster",
                compare_to_ground_truth=True,
                results_file=cluster_results_file,
                dataset=dataset,
                method=graph_type,
                time_reports=times,
            )

        # Eval cluster against brute force DPC
        eval_cluster_and_write_results(
            gt_cluster_path=f"results/{dataset_folder}/{dataset}_BruteForce.cluster",
            cluster_path=f"{prefix}.cluster",
            compare_to_ground_truth=False,
            results_file=cluster_results_file,
            dataset=dataset,
            method=graph_type,
            time_reports=times,
        )

    for graph_type, command in tqdm(options):
        new_framework_settings = []
        if run_new_dpc_framework:
            new_framework_settings.append(True)
        if run_old_dpc_framework:
            new_framework_settings.append(False)
        for use_new_framework in new_framework_settings:
            p = multiprocessing.Process(
                target=try_command, args=(graph_type, command, use_new_framework)
            )
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
    )

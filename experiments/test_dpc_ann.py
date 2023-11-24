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


def create_density_computer(density_info, data):
    if density_info == "kth":
        return dpc_ann.KthDistanceDensityComputer()
    if density_info == "normalized":
        return dpc_ann.NormalizedDensityComputer()
    if density_info == "race":
        # These numbers are just a really basic heuristic
        return dpc_ann.RaceDensityComputer(
            num_estimators=32,
            hashes_per_estimator=18,
            data_dim=data.shape[1],
            lsh_family=dpc_ann.CosineFamily(),
        )
    if density_info == "exp-sum":
        return dpc_ann.ExpSquaredDensityComputer()
    if density_info == "sum-exp":
        return dpc_ann.SumExpDensityComputer()
    if density_info == "sum":
        return dpc_ann.TopKSumDensityComputer()
    if density_info == "mutual":
        return dpc_ann.MutualKNNDensityDensityComputer()
    raise ValueError(f"Unknown density type {density_info}")


def get_configurations_to_run(
    dataset,
    num_clusters,
    graph_types,
    search_range,
    compare_against_bf,
    density_methods,
    Ks,
    dataset_folder,
    data,
):
    configurations = []

    alpha = 1.1  # For now just always use alpha = 1.1, seems to perform the best

    for K in Ks:
        for density_method in density_methods:
            ground_truth_cluster_path = f"results/{dataset_folder}/{dataset}_BruteForce_{density_method}_{K}.cluster"
            run_bf = graph_types[0] == "BruteForce" or (
                compare_against_bf and not os.path.isfile(ground_truth_cluster_path)
            )
            if run_bf:
                command_line = {
                    "graph_type": "BruteForce",
                    "output_path": ground_truth_cluster_path,
                    "density_computer": create_density_computer(density_method, data),
                    "center_finder": dpc_ann.ProductCenterFinder(
                        num_clusters=num_clusters,
                        use_reweighted_density=(density_method == "normalized"),
                    ),
                    "K": K,
                }
                configurations.append(
                    (f"BruteForce_{density_method}_{K}", command_line)
                )

            for graph_type in graph_types:
                if graph_type == "BruteForce":
                    continue

                for (
                    max_degree,
                    beam_search_construction,
                    beam_search_clustering,
                    beam_search_density,
                ) in itertools.product(
                    search_range, search_range, search_range, search_range
                ):
                    if graph_type in ["pyNNDescent", "HCNNG"] and (
                        beam_search_construction < 16
                    ):
                        continue
                    
                    method = f"{graph_type}_{max_degree}_{alpha}_{beam_search_construction}_{beam_search_density}_{beam_search_clustering}_{density_method}_{K}"
                    command_line = {
                        "max_degree": max_degree,
                        "alpha": alpha,
                        "Lbuild": beam_search_construction,
                        "L": max(2 * K, beam_search_density),
                        "Lnn": beam_search_clustering,
                        "graph_type": graph_type,
                        "density_computer": create_density_computer(
                            density_method, data
                        ),
                        "center_finder": dpc_ann.ProductCenterFinder(
                            num_clusters=num_clusters,
                            use_reweighted_density=(density_method == "normalized"),
                        ),
                        "K": K,
                    }
                    if graph_type in ["pyNNDescent", "HCNNG"]:
                        num_clusters_in_build = 1  # For now just always use 1 cluster, seems to perform the best
                        command_line["num_clusters"] = num_clusters_in_build
                        # TODO: For now not recording num clusters in graph building because always choosing 1
                        # method += "_" + str(num_clusters_in_build)

                    configurations.append((method, command_line))

    return configurations


def run_dpc_ann_configurations(
    dataset,
    timeout_s,
    num_clusters,
    graph_types=["Vamana", "HCNNG", "pyNNDescent"],
    search_range="Default",  # Default is [8, 16, 32, 64] unless len(data) > 250,000, then also 128 and 256
    compare_against_gt=True,
    compare_against_bf=True,
    density_methods=["kth"],
    Ks=[6],
    results_file_prefix="",
):
    cluster_results_file = create_results_file(prefix=results_file_prefix)

    dataset_folder = make_results_folder(dataset)
    data = np.load(f"data/{dataset_folder}/{dataset}.npy").astype("float32")

    if search_range == "Default":
        search_range = [8, 16, 32, 64]
        if len(data) > 250000:
            search_range += [128, 256]

    if "BruteForce" in graph_types and graph_types[0] != "BruteForce":
        raise ValueError("If running BruteForce, BruteForce must come first in list.")

    configurations = get_configurations_to_run(
        dataset,
        num_clusters,
        graph_types,
        search_range,
        compare_against_bf,
        density_methods,
        Ks,
        dataset_folder,
        data,
    )

    def try_command(graph_type, command):
        prefix = f"results/{dataset_folder}/{dataset}_{graph_type}"

        clustering_result = dpc_ann.dpc_numpy(
            **command,
            data=data,
            decision_graph_path=f"{prefix}.dg",
        )

        # Eval cluster against ground truth and write results
        if compare_against_gt:
            eval_cluster_and_write_results(
                gt_cluster_path=f"data/{dataset_folder}/{dataset}.gt",
                found_clusters=np.array(clustering_result.clusters),
                comparing_to_ground_truth=True,
                results_file=cluster_results_file,
                dataset=dataset,
                method=graph_type,
                time_reports=clustering_result.metadata,
            )

        # Eval cluster against brute force DPC
        if compare_against_bf:
            bf_extension = "_".join(graph_type.split("_")[-2:])
            eval_cluster_and_write_results(
                gt_cluster_path=f"results/{dataset_folder}/{dataset}_BruteForce_{bf_extension}.cluster",
                found_clusters=np.array(clustering_result.clusters),
                comparing_to_ground_truth=False,
                results_file=cluster_results_file,
                dataset=dataset,
                method=graph_type,
                time_reports=clustering_result.metadata,
            )

    for graph_type, command in tqdm(configurations):
        p = multiprocessing.Process(target=try_command, args=(graph_type, command))
        p.start()

        if graph_type.startswith("BruteForce"):
            exitcode = p.join()  # No timeout when doing brute force
        else:
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
    parser.add_argument("--dont_compare_against_bf", default=False, action="store_true")
    parser.add_argument("-search_range", nargs="+", type=int)
    parser.add_argument("-graph_types", nargs="+", type=str)
    parser.add_argument("-Ks", nargs="+", type=int)
    parser.add_argument("-density_methods", nargs="+", type=str)
    args = parser.parse_args()

    extras = {}
    arg_dict = vars(args)
    for optional in ["search_range", "graph_types", "Ks", "density_methods"]:
        if arg_dict[optional] != None:
            extras[optional] = arg_dict[optional]

    run_dpc_ann_configurations(
        dataset=args.dataset,
        timeout_s=args.timeout,
        num_clusters=args.num_clusters,
        compare_against_gt=not args.dont_compare_against_gt,
        compare_against_bf=not args.dont_compare_against_bf,
        results_file_prefix=args.results_file_prefix,
        **extras,
    )

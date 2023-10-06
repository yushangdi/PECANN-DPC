#!/usr/bin/env python3

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
    get_cutoff,
)

import dpc_ann


def run_basic_experiments(datasets=["s2", "s3", "unbalance"]):
    cluster_results_file = create_results_file()

    for dataset in datasets:
        dataset_folder = make_results_folder(dataset)
        for graph_type in ["BruteForce", "HCNNG", "pyNNDescent", "Vamana"]:
            query_file = f"data/{dataset_folder}/{dataset}.txt"
            prefix = f"results/{dataset_folder}/{dataset}_{graph_type}"

            clustering_result = dpc_ann.dpc_filenames(
                data_path=query_file,
                decision_graph_path=f"{prefix}.dg ",
                output_path=f"{prefix}.cluster",
                graph_type=graph_type,
                **get_cutoff(dataset),
            )

            time_reports = clustering_result.metadata

            # Eval cluster against ground truth and write results
            eval_cluster_and_write_results(
                gt_cluster_path=f"data/{dataset_folder}/{dataset}.gt",
                cluster_path=f"{prefix}.cluster",
                compare_to_ground_truth=True,
                results_file=cluster_results_file,
                dataset=dataset,
                method=graph_type,
                time_reports=time_reports,
            )

            # Eval cluster against brute force DPC
            eval_cluster_and_write_results(
                gt_cluster_path=f"results/{dataset_folder}/{dataset}_BruteForce.cluster",
                cluster_path=f"{prefix}.cluster",
                compare_to_ground_truth=False,
                results_file=cluster_results_file,
                dataset=dataset,
                method=graph_type,
                time_reports=time_reports,
            )

            # Plot clusters
            plot_dims(
                filename=query_file,
                cluster_path=f"{prefix}.cluster",
                image_path=f"{prefix}.png",
            )


if __name__ == "__main__":
    run_basic_experiments()

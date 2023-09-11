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
    get_cutoff,
)

import dpc_ann

from contextlib import redirect_stdout
import io


cluster_results_file = create_results_file()

for dataset in ["s2", "s3", "unbalance", "mnist"]:
    dataset_folder = make_results_folder(dataset)
    for method in ["BruteForce", "HCNNG", "pyNNDescent", "Vamana"]:
        query_file = f"data/{dataset_folder}/{dataset}.txt"
        prefix = f"results/{dataset_folder}/{dataset}_{method}"

        # TODO: Return times instead of capturing output
        stdout = io.StringIO()
        with redirect_stdout(f):
            dpc_ann.dpc(
                query_file=query_file,
                decision_graph_path=f"{prefix}.dg ",
                output_file=f"{prefix}.cluster ",
                graph_type={method},
                **get_cutoff(dataset),
            )

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

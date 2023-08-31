#!/usr/bin/env python3

import os
from pathlib import Path

from data_processors.plot import plot_dims
from post_processors.cluster_eval import eval_cluster

# Change to DPC-ANN folder
abspath = Path(__file__).resolve().parent
os.chdir(abspath)

cluster_results_file = "results/cluster_analysis.csv"
headers = ["recall50", "precision50", "AMI", "ARI", "completeness", "homogeneity"]
with open(cluster_results_file, "w") as f:
    f.write("dataset,method," + ",".join(headers))


distance_cutoff_map = {"s2": 102873, "s3": 102873, "unbalance": 30000}
num_cluster_map = {"s2": 15, "s3": 15, "unbalance": 8}

for dataset in ["s2", "s3", "unbalance"]:
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
            + f"--dist_cutoff {distance_cutoff_map[dataset]} "
        )
        if method == "bruteforce":
            dpc_command += f"--bruteforce true "
        # else:
        #     dpc_command += f"--graph_type {method}"

        # Run DPC
        os.system(dpc_command)

        # Eval cluster
        cluster_results = eval_cluster(gt_path=f"data/{dataset_folder}/{dataset}-label.pa", cluster_path=f"{prefix}.cluster")

        with open(cluster_results_file, "a") as f:
            fields = [dataset, method] + [str(cluster_results[h]) for h in headers]
            f.write(",".join(fields) + "\n")


        # Plot clusters
        plot_dims(filename=query_file, cluster_path=f"{prefix}.cluster", image_path=f"{prefix}.png")




# # python3 post_processors/plot_decision_graph.py results/mnist_bruteforce.dg 10 mnist_bruteforce
# # python3 post_processors/cluster_eval.py ./data/mnist.gt results/mnist_bruteforce.cluster

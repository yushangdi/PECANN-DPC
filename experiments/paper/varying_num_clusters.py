import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np


abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from test_dpc_ann import run_dpc_ann_configurations
from utils import product_cluster_dg

os.chdir(abspath.parent)
sys.path.append(str(abspath.parent))
from post_processors.cluster_eval import eval_clusters

results_file = "results/cluster_analysis_varying_num_clusters.csv"
with open(results_file, "w") as f:
    f.write("dataset,num_clusters,ARI\n")

results = []
for dataset, num_gt_clusters, param_value in [
    ("mnist", 10, 32),
    ("imagenet", 1000, 128),
    ("arxiv-clustering-s2s", 180, 64),
    ("reddit-clustering", 50, 64),
    ("birds", 525, 32),
]:
    dg_path = f"results/{dataset}/{dataset}_Vamana_{param_value}_1.1_{param_value}_{param_value}_{param_value}_kth_16.dg"
    if not os.path.exists(dg_path):
        run_dpc_ann_configurations(
            dataset,
            timeout_s=2000,
            num_clusters=num_gt_clusters,
            graph_types=["Vamana"],
            search_range=[param_value],
            compare_against_bf=False,
            density_methods=["kth"],
            Ks=[16],
        )

    ground_truth = np.loadtxt(f"data/{dataset}/{dataset}.gt", dtype=int)

    def get_ari(clusters):
        return eval_clusters(
            ground_truth, clusters, verbose=False, eval_metrics=["ARI"]
        )["ARI"]

    num_cluster_values = set(
        range(1, 10 * num_gt_clusters + 1, (10 * num_gt_clusters + 1) // 100)
    )

    results = product_cluster_dg(
        dg_path, num_cluster_values=num_cluster_values, callback=get_ari
    )
    results = results[::-1]

    with open(results_file, "a") as f:
        for num_clusters, ari in results:
            f.write(f"{dataset},{num_clusters},{ari}\n")

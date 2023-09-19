import numpy as np
import sys
from collections import Counter
import os
from pathlib import Path

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from post_processors.cluster_eval import eval_clusters
from utils import product_cluster_dg

# dataset = "mnist"
# gt_num_clusters = 10
# num_clusters_to_test = range(10, 50)

dataset = "imagenet"
gt_num_clusters = 1000
num_clusters_to_test = [1000, 1200, 1400]

ground_truth = np.loadtxt(f"data/{dataset}/{dataset}.gt", dtype=int)

dg_path = f"results/{dataset}/{dataset}_BruteForce.dg"

for num_clusters in num_clusters_to_test:
    clusters = product_cluster_dg(dg_path, num_clusters=num_clusters)

    eval_result = eval_clusters(ground_truth, clusters, verbose=False, metrics=["ari"])
    cluster_counts = Counter(clusters)

    num_non_one_clusters = len(
        [count for count in cluster_counts.values() if count > 1]
    )
    cutoff = min([b for _, b in cluster_counts.most_common(gt_num_clusters)])

    print(num_clusters, eval_result["ARI"], cutoff)

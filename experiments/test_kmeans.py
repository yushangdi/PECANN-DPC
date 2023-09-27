import faiss
import numpy as np
import sys
import os
from pathlib import Path
import torch
import time

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from post_processors.cluster_eval import eval_clusters
from utils import create_results_file, eval_cluster_and_write_results

x = np.load("data/imagenet/imagenet.npy")
ncentroids = 1000

results_file = create_results_file("kmeans")

for nredo in range(1, 5):
    for niter in list(range(1, 10)) + list(range(10, 50, 5)):
        if nredo * niter > 100:
            continue
        
        built_start_time = time.time()
        verbose = False
        d = x.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(x)
        built_time = time.time() - built_start_time

        find_clusters_start = time.time()
        batch_size = 100000
        clusters = []
        for start in range(0, len(x), batch_size):
            batch_x = x[start : start + batch_size]
            distances = torch.cdist(torch.tensor(batch_x), torch.tensor(kmeans.centroids))
            clusters += torch.argmin(distances, dim=1).tolist()
        find_clusters_time = time.time() - find_clusters_start

        total_time = time.time() - built_start_time
        with open("results/kmeans.cluster", "w") as f:
            for i in clusters:
                f.write(str(i) + "\n")

        method_name = f"kmeans_{niter}_{nredo}"
        times = {
            "Total": total_time,
            "Find clusters": find_clusters_time,
            "Built index": built_time,
        }

        eval_cluster_and_write_results(
            gt_cluster_path=f"data/imagenet/imagenet.gt",
            cluster_path=f"results/kmeans.cluster",
            compare_to_ground_truth=True,
            results_file=results_file,
            dataset="imagenet",
            method=method_name,
            time_reports=times,
        )

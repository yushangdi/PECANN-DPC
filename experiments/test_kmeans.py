import faiss
import numpy as np
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import torch
import time
import argparse

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from utils import (
    create_results_file,
    eval_cluster_and_write_results,
    make_results_folder,
)


def run_kmeans(data, num_clusters, nredo, niter):
    built_start_time = time.time()
    verbose = True
    d = data.shape[1]
    kmeans = faiss.Kmeans(d, num_clusters, niter=niter, nredo=nredo, verbose=verbose)
    kmeans.train(data)
    built_time = time.time() - built_start_time

    find_clusters_start = time.time()
    batch_size = 100000
    clusters = []
    for start in range(0, len(data), batch_size):
        batch_x = data[start : start + batch_size]
        distances = torch.cdist(torch.tensor(batch_x), torch.tensor(kmeans.centroids))
        clusters += torch.argmin(distances, dim=1).tolist()
    find_clusters_time = time.time() - find_clusters_start

    return clusters, built_time, find_clusters_time, time.time() - built_start_time


def run_kmeans_experiment(
    dataset,
    num_clusters=None,
    nredo_options=list(range(1, 5)),
    niter_options=list(range(1, 10)) + list(range(10, 50, 5)),
    max_iterations=100,
    prefix="kmeans",
):
    if num_clusters is None:
        num_clusters = int(len(set(np.loadtxt(f"data/{dataset}/{dataset}.gt"))))

    print(f"Clustering with {num_clusters} clusters")

    x = np.load(f"data/{dataset}/{dataset}.npy")
    make_results_folder(dataset)
    cluster_result_path = f"results/{dataset}/kmeans.cluster"
    results_file = create_results_file(prefix)

    for nredo in nredo_options:
        for niter in niter_options:
            if nredo * niter > max_iterations:
                continue

            clusters, built_time, find_clusters_time, total_time = run_kmeans(
                x, num_clusters, nredo, niter
            )

            with open(cluster_result_path, "w") as f:
                for i in clusters:
                    f.write(str(i) + "\n")

            method_name = f"kmeans_{niter}_{nredo}"
            times = {
                "Total": total_time,
                "Find clusters": find_clusters_time,
                "Built index": built_time,
            }

            eval_cluster_and_write_results(
                gt_cluster_path=f"data/{dataset}/{dataset}.gt",
                found_clusters=cluster_result_path,
                comparing_to_ground_truth=True,
                results_file=results_file,
                dataset=dataset,
                method=method_name,
                time_reports=times,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run kmeans on passed in dataset.")
    parser.add_argument(
        "dataset",
        help="Dataset name (should be a file named data/<dataset>/<dataset>.npy and data/dataset/<dataset>gt).",
    )
    parser.add_argument(
        "--num_clusters",
        help="How many clusters to use for kmeans (default is the exact number of classes in the ground truth)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_iterations",
        help="The maximum number of iterations (across nredo and niter) to use for kmeans",
        default=100,
        type=int,
    )
    args = parser.parse_args()

    run_kmeans_experiment(args.dataset, args.num_clusters.args.max_iterations)

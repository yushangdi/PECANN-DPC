import numpy as np
import sys
import os
from pathlib import Path
import time
import argparse
from fastdp import fastdp

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from utils import (
    create_results_file,
    eval_cluster_and_write_results,
    make_results_folder,
)


parser = argparse.ArgumentParser(description="Run fastdp on passed in dataset.")
parser.add_argument(
    "dataset",
    help="Dataset name (should be a file named data/<dataset>/<dataset>.npy and data/dataset/<dataset>gt).",
)
parser.add_argument(
    "--K",
    help="How many neighbors the k nearest neighbor graph has per point",
    type=int,
    default=16,
)
parser.add_argument(
    "--num_clusters",
    help="How many clusters to use (default is the exact number of classes in the ground truth)",
    type=int,
    default=None,
)
parser.add_argument(
    "--window",
    help="Beam search buffer size?",
    type=int,
    default=50,
)


args = parser.parse_args()

num_clusters = args.num_clusters
if num_clusters is None:
    num_clusters = int(len(set(np.loadtxt(f"data/{args.dataset}/{args.dataset}.gt"))))

print(f"Clustering with {num_clusters} clusters")

x = np.load(f"data/{args.dataset}/{args.dataset}.npy")
make_results_folder(args.dataset)
cluster_result_path = f"results/{args.dataset}/fastdp.cluster"
results_file = create_results_file("fastdp")

window = args.window
for maxiter in [1, 2, 4, 8, 16, 32, 64]:
    start = time.time()
    (clusters, peak_ids) = fastdp(
        x,
        num_clusters,
        distance="l2",
        num_neighbors=args.K,
        window=window,
        nndes_start=0.2,
        maxiter=maxiter,
        endcond=0,
        dtype="vec",
    )

    times = {"Total": time.time() - start}

    with open(cluster_result_path, "w") as f:
        for i in clusters:
            f.write(str(i) + "\n")

    eval_cluster_and_write_results(
        gt_cluster_path=f"data/{args.dataset}/{args.dataset}.gt",
        found_clusters=cluster_result_path,
        comparing_to_ground_truth=True,
        results_file=results_file,
        dataset=args.dataset,
        method=f"fastdp_{window}_{maxiter}_{args.K}",
        time_reports=times,
    )

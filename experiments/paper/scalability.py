import dpc_ann
import os
from pathlib import Path
import sys
import numpy as np
from paper_utils import get_core_groups, generate_synthetic_data

abspath = Path(__file__).resolve().parent.parent
sys.path.append(str(abspath))
sys.path.append(str(abspath.parent))
os.chdir(abspath.parent)

from utils import (
    create_results_file,
    eval_cluster_and_write_results,
    make_results_folder,
)

initial_threads = os.sched_getaffinity(0)

cluster_results_file = create_results_file(prefix="scalability")
dataset_folder = make_results_folder("synthetic")

num_datapoints_to_cluster = [10**5, 10**6, 10**7, 10**8, 10**9]
num_clusters = 100

current_threads = []
core_groups = get_core_groups()


def flatten(l):
    return [item for sublist in l for item in sublist]


threads = flatten(core_groups)

print(threads)
for thread in threads:
    current_threads.append(thread)
    if len(current_threads) not in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        continue
    os.environ["PARLAY_NUM_THREADS"] = f"{len(current_threads)}"
    print(f"Running experiment with {len(current_threads)} threads: {current_threads}")

    # TODO: Do we want to do something to ensure accuracy is above a certain threshold?
    for num_datapoints in num_datapoints_to_cluster:
        predicted_time = num_datapoints / 4000 / len(current_threads)
        if predicted_time > 10000:
            break

        os.sched_setaffinity(0, initial_threads)

        data, gt, best_ari = generate_synthetic_data(num_datapoints, num_clusters)

        os.sched_setaffinity(0, current_threads)

        gt_path = f"results/{dataset_folder}/temp.gt"
        np.savetxt(gt_path, gt, fmt="%i")

        clustering_result = dpc_ann.dpc_numpy(
            data=data,
            max_degree=32,
            alpha=1.1,
            Lbuild=32,
            L=32,
            Lnn=32,
            graph_type="Vamana",
            density_computer=dpc_ann.KthDistanceDensityComputer(),
            center_finder=dpc_ann.ProductCenterFinder(num_clusters=num_clusters),
            K=16,
        )

        eval_cluster_and_write_results(
            gt_cluster_path=gt_path,
            found_clusters=np.array(clustering_result.clusters),
            compare_to_ground_truth=False,
            results_file=cluster_results_file,
            dataset="synthetic",
            method="Vamana_standard",
            time_reports=clustering_result.metadata,
        )

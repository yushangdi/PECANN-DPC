import os
from pathlib import Path
import sys
import numpy as np
from paper_utils import generate_synthetic_data
import dpc_ann

abspath = Path(__file__).resolve().parent.parent
sys.path.append(str(abspath))
sys.path.append(str(abspath.parent))
os.chdir(abspath.parent)

from utils import (
    create_results_file,
    eval_cluster_and_write_results,
    make_results_folder,
)

num_datapoints_to_cluster = [10**5, 10**6, 10**7, 10**8, 10**9]
num_clusters_options = [10**1, 10**2, 10**3, 10**4]

dataset_folder = make_results_folder("synthetic")


def run_synthetic():
    cluster_results_file = create_results_file(prefix="thread_scaling")

    for num_datapoints in num_datapoints_to_cluster:
        for num_clusters in num_clusters_options:
            data, gt = generate_synthetic_data(num_datapoints, num_clusters)

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
                comparing_to_ground_truth=True,
                results_file=cluster_results_file,
                dataset=f"synthetic_{num_datapoints}_{num_clusters}",
                method="Vamana_standard",
                time_reports=clustering_result.metadata,
            )


if __name__ == "__main__":
    run_synthetic()

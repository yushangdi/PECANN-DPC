import os
from pathlib import Path
import sys
import numpy as np
import dpc_ann
import multiprocessing

from sklearn.utils import shuffle


def generate_synthetic_data(num_datapoints, num_clusters, d=128, variance=0.05):
    points_per_cluster = num_datapoints // num_clusters
    cluster_centers = np.random.uniform(size=(num_clusters, d))
    data = []
    gts = []

    for i, center in enumerate(cluster_centers):
        data.append(
            np.random.multivariate_normal(
                mean=center, cov=np.eye(d) * variance, size=points_per_cluster
            )
        )
        gts += [i] * points_per_cluster

    data = np.vstack(data)
    data = data.astype("float32")
    gt = np.array(gts, dtype=int)
    shuffle(data, gt, random_state=0)

    return data, gt


abspath = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(abspath))
sys.path.append(str(abspath.parent))
os.chdir(abspath.parent)

from utils import (
    create_results_file,
    eval_cluster_and_write_results,
    make_results_folder,
)

num_datapoints_to_cluster = [10**5, 10**6, 10**7, 10**8, 4 * 10**8]
num_clusters_options = [10**1, 10**2, 10**3, 10**4]

dataset_folder = make_results_folder("synthetic")


def run_synthetic_experiment(num_datapoints, num_clusters, cluster_results_file):
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


def run_synthetic():
    cluster_results_file = create_results_file(prefix="dataset_size_scaling")

    for num_datapoints in num_datapoints_to_cluster:
        for num_clusters in num_clusters_options:
            p = multiprocessing.Process(
                target=run_synthetic_experiment,
                args=(num_datapoints, num_clusters, cluster_results_file),
            )
            p.start()
            p.join()


if __name__ == "__main__":
    run_synthetic()

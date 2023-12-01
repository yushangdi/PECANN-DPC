import os
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

abspath = Path(__file__).resolve().parent.parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from test_dpc_ann import run_dpc_ann_configurations
from utils import product_cluster_dg
from test_kmeans import run_kmeans

os.chdir(abspath.parent)
sys.path.append(str(abspath.parent))
from post_processors.cluster_eval import eval_clusters


def get_kmeans_clustering_results(dataset, num_cluster_values, get_ari_func):
    dataset = np.load(f"data/{dataset}/{dataset}.npy")
    results = []
    for num_clusters in tqdm(num_cluster_values):
        clusters, _, _, _ = run_kmeans(dataset, num_clusters, nredo=1, niter=20)
        clusters = np.array(clusters)
        print(clusters)
        results.append((num_clusters, get_ari_func(clusters)))
    return results


def get_vamana_clustering_results(
    dataset, num_cluster_values, get_ari_func, vamana_param_value
):
    dg_path = f"results/{dataset}/{dataset}_Vamana_{vamana_param_value}_1.1_{vamana_param_value}_{vamana_param_value}_{vamana_param_value}_kth_16.dg"
    if not os.path.exists(dg_path):
        run_dpc_ann_configurations(
            dataset,
            timeout_s=2000,
            num_clusters=1,
            graph_types=["Vamana"],
            search_range=[vamana_param_value],
            compare_against_bf=False,
            density_methods=["kth"],
            Ks=[16],
        )

    results = product_cluster_dg(
        dg_path, num_cluster_values=num_cluster_values, callback=get_ari_func
    )

    return results[::-1]


def run_varying_num_clusters_vamana_and_kmeans():
    results_file = "results/cluster_analysis_varying_num_clusters.csv"
    eval_metrics = ["ARI", "homogeneity", "completeness", "recall50", "precision50"]
    with open(results_file, "w") as f:
        f.write("method,dataset,num_clusters," + ",".join(eval_metrics) + "\n")

    for dataset, num_gt_clusters, vamana_param_value in [
        ("mnist", 10, 32),
        ("imagenet", 1000, 128),
        ("arxiv-clustering-s2s", 180, 64),
        ("reddit-clustering", 50, 64),
        ("birds", 525, 32),
    ]:
        num_cluster_values = set(
            range(1, 10 * num_gt_clusters + 1, (10 * num_gt_clusters + 1) // 100)
        )

        ground_truth = np.loadtxt(f"data/{dataset}/{dataset}.gt", dtype=int)

        def get_ari(clusters):
            return eval_clusters(
                ground_truth,
                clusters,
                verbose=False,
                eval_metrics=[
                    "ARI",
                    "homogeneity",
                    "completeness",
                    "recall50",
                    "precision50",
                ],
            )

        vamana_results = get_vamana_clustering_results(
            dataset, num_cluster_values, get_ari, vamana_param_value
        )

        kmeans_results = get_kmeans_clustering_results(
            dataset, num_cluster_values, get_ari
        )

        with open(results_file, "a") as f:
            for num_clusters, result_dict in vamana_results:
                f.write(
                    f"Vamana,{dataset},{num_clusters},{','.join([str(result_dict[metric]) for metric in eval_metrics])}\n"
                )
            for num_clusters, result_dict in kmeans_results:
                f.write(
                    f"kmeans,{dataset},{num_clusters},{','.join([str(result_dict[metric]) for metric in eval_metrics])}\n"
                )


if __name__ == "__main__":
    run_varying_num_clusters_vamana_and_kmeans()

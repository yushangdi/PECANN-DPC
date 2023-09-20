import numpy as np
import pandas as pd
from experiments.union_find import UnionFind
from post_processors.cluster_eval import eval_clusters
import sys
from collections import Counter
import os
from pathlib import Path

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

dataset = "mnist"
gt_num_clusters = 10
num_clusters_to_test = range(10, 50)

# dataset = "imagenet"
# gt_num_clusters = 1000
# num_clusters_to_test = [1000, 2000, 4000, 8000, 16000, 32000]

ground_truth = np.loadtxt(f"data/{dataset}/{dataset}.gt", dtype=int)

graph = pd.read_csv(f"results/{dataset}/{dataset}_Vamana.dg", delimiter=" ")

# Force parentless nodes to be selected by any strategy
parentless = graph["Parent_Distance"] == -1
graph.loc[parentless, "Density"] = 1
graph.loc[parentless, "Parent_Distance"] = float("inf")

# Force duplicates to be never selected by any strategy
duplicates = graph["Parent_Distance"] == 0
graph.loc[duplicates, "Parent_Distance"] = sys.float_info.min


def cluster_by_densities_distance_product(
    graph, num_clusters, density_product, distance_product
):
    parents = graph["Parent_ID"].to_numpy(copy=True)
    new_column = (np.log(graph["Density"]) * density_product) + (
        np.log(graph["Parent_Distance"]) * distance_product
    )
    top_k_densities = new_column.nlargest(num_clusters)
    parents[top_k_densities.index] = -1

    u = UnionFind()
    for i, p in enumerate(parents):
        if p != -1:
            u.unite(i, p)
        else:
            u.add(i)

    return np.array([u.find(i) for i in range(len(graph))])


for num_clusters in num_clusters_to_test:
    clusters = cluster_by_densities_distance_product(
        graph, num_clusters=num_clusters, density_product=1, distance_product=1
    )

    eval_result = eval_clusters(ground_truth, clusters, verbose=False, eval_metrics=["ARI"])
    cluster_counts = Counter(clusters)

    num_non_one_clusters = len(
        [count for count in cluster_counts.values() if count > 1]
    )
    cutoff = min([b for _, b in cluster_counts.most_common(gt_num_clusters)])

    print(num_clusters, eval_result["ARI"], cutoff)

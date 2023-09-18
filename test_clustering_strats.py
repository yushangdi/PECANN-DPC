import numpy as np
import pandas as pd
from union_find import UnionFind
from post_processors.cluster_eval import eval_clusters

ground_truth = np.loadtxt("data/imagenet/imagenet.gt", dtype=int)

graph = pd.read_csv("results/imagenet/imagenet_Vamana.dg", delimiter=" ")

def cluster_by_densities_distance_product(graph, num_clusters, density_product, distance_product):
    parents = graph["Parent_ID"].to_numpy(copy=True)
    new_column = (np.log(graph["Density"]) * density_product) + (np.log(graph["Parent_Distance"]) * distance_product)
    top_k_densities = new_column.nlargest(num_clusters)
    parents[top_k_densities.index] = -1

    u = UnionFind()
    for i, p in enumerate(parents):
        if p != -1:
            u.unite(i, p)
        else:
            u.add(i)

    return np.array([u.find(i) for i in range(len(graph))])


for num_clusters in [1000, 2000, 4000, 8000, 16000]:
    by_highest_density = cluster_by_densities_distance_product(graph, num_clusters=num_clusters, density_product=1, distance_product=1)
    print(num_clusters, eval_clusters(ground_truth, by_highest_density, verbose=False)["ARI"], flush=True)
# print(by_highest_density[:5])
# print(ground_truth[:5])
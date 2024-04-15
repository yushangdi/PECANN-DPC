# from dbscan import DBSCAN
# from sklearn.datasets import make_blobs


# points, y = make_blobs(n_samples=20, centers=3, n_features=784, random_state=0)

# labels, core_samples_mask = DBSCAN(points, eps=0.3, min_samples=10)

import numpy as np
from sklearn.cluster import DBSCAN
import sklearn
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import time



# print(np.where(data!=0))
# print(data[np.where(data!=0)])
# epsilon chosen 


# for epsilon > 8, number cluster is 1.
# for eps in [2,3,4,5,6,7,8]:
#   for min_samples in [1,2,3,4,5]:
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
#     print("num clusters, ", len(np.unique(clustering.labels_)))
#     print("num noise, ", np.sum(clustering.labels_==-1))
#     ari = sklearn.metrics.adjusted_rand_score(labels, clustering.labels_)
#     print(eps,  min_samples, ari)

## when epsilon =1, min sample > 2, everything is noise


# "arxiv-clustering-s2s": "arxiv",
#     "reddit-clustering": "reddit",
#     "imagenet": "ImageNet",
#     "mnist": "MNIST",
#     "birds": "birds",

dataset = "mnist"
data = np.load(f"/home/sy/embeddings/{dataset}/{dataset}.npy").astype("float32")
labels = np.loadtxt(f"/home/sy/embeddings/{dataset}/{dataset}.gt").flatten()

print(dataset, "data loaded")

#"mnist"
# eps_values = [3, 4, 5, 6, 7, 8, 9]
# min_samples_values = [1, 2 , 3, 4, 5, 10, 50, 100, 300, 500, 600, 700, 800, 900, 1000, 5000, 6000, 7000, 8000, 9000]
# birds
# > 30 too few clusters
# 5, 10, 15, 20, 25
# eps_values = [6,7,8,9,11,12,13,14]
# min_samples_values = [1, 2, 3, 4, 5, 120, 150, 180, 210, 240, 270]
# "reddit-clustering"
# eps_values = [0.4, 0.46, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66]
# min_samples_values = [1, 2, 3, 4, 5]

# Initialize an empty list to store results
results = []

# Iterate over parameter combinations
for eps in eps_values:
    for min_samples in min_samples_values:
        # Perform DBSCAN clustering
        start = time.time()
        # on high dimensional data, bruteforce is faster than kd-tree and ball-tree
        clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm="brute", n_jobs=60).fit(data)
        clustering_time = time.time() - start
        # Compute evaluation metric (e.g., adjusted Rand index)
        ari = sklearn.metrics.adjusted_rand_score(labels, clustering.labels_)
        
        num_clusters = len(np.unique(clustering.labels_))
        num_noise = np.sum(clustering.labels_ == -1)
        # Append results to the list
        results.append({'dataset': dataset,
                        'eps': eps,
                        'min_samples': min_samples,
                        'num_clusters': num_clusters ,
                        'num_noise': num_noise,
                        'ARI': ari,
                        "sklearn_time": clustering_time})
        print(eps, min_samples, ari, clustering_time)
        print(num_clusters, num_noise)

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Print or further analyze the results
print(results_df)
results_df.to_csv(f"../results/{dataset}2.csv")
from dbscan import DBSCAN
# from sklearn.datasets import make_blobs
# points, y = make_blobs(n_samples=20, centers=3, n_features=784, random_state=0)
# labels, core_samples_mask = DBSCAN(points, eps=0.3, min_samples=10)

import numpy as np
import sklearn
import sklearn.metrics
import pandas as pd
import time
import matplotlib.pyplot as plt

# dataset = "unbalance"
# data = np.loadtxt(f"../data/{dataset}/{dataset}.txt").astype("float32")
# labels = np.loadtxt(f"../data/{dataset}/{dataset}.gt").flatten()
dataset = "s2"
data = np.loadtxt(f"../data/s_datasets/{dataset}.txt").astype("float32")
labels = np.loadtxt(f"../data/s_datasets/{dataset}.gt").flatten()

print(dataset, "data loaded")


# Unblance
# eps_values = [3000, 5000, 4000, 6000, 10000]
# min_samples_values = [10, 25, 50, 100]

# S2
eps_values = [75000, 80000] #,85000, 90000
min_samples_values = [200, 250, 300, 350]
for i in range(220, 270):
   min_samples_values.append(i)


# Initialize an empty list to store results
results = []

# Iterate over parameter combinations
for eps in eps_values:
    for min_samples in min_samples_values:
        # Perform DBSCAN clustering
        start = time.time()
        # on high dimensional data, bruteforce is faster than kd-tree and ball-tree
        clustering, core_samples_mask = DBSCAN(data, eps=eps, min_samples=min_samples)
        # clustering = clustering.astype(str)
        clustering_time = time.time() - start
        # Compute evaluation metric (e.g., adjusted Rand index)
        ari = sklearn.metrics.adjusted_rand_score(labels, clustering)
        
        num_clusters = len(np.unique(clustering))
        num_noise = np.sum(clustering == -1)
        # Append results to the list
        results.append({'dataset': dataset,
                        'eps': eps,
                        'min_samples': min_samples,
                        'num_clusters': num_clusters ,
                        'num_noise': num_noise,
                        'ARI': ari,
                        "time": clustering_time})
        # print(eps, min_samples, ari, clustering_time)
        # print(num_clusters, num_noise)
        plt.clf()
        plt.figure(figsize=(10, 6))
        # Plot points
        plt.scatter(data[:, 0], data[:, 1], c=clustering, cmap='viridis', s=50, alpha=0.7)

        # Add color bar
        plt.colorbar(label='Cluster Label')

        plt.title(f'DBSCAN Clustering {eps}, {min_samples}')
        plt.savefig(f'DBSCAN-{eps}-{min_samples}.png')
        plt.close()

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Print or further analyze the results
print(results_df)
results_df.to_csv(f"../results/{dataset}.csv")
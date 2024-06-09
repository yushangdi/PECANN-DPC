#  pip3 install install scikit-learn-intelex --break-system-packages
# https://github.com/intel/scikit-learn-intelex
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import DBSCAN
import sklearn
import pandas as pd
import time

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

# "unbalance", "s2", 
datasets = ["mnist", "birds", "arxiv-clustering-s2s", "imangenet", "reddit-clustering"]

for dataset in datasets:
    if dataset == "unbalance":
        data = np.loadtxt(f"/home/sy/PECANN-DPC/data/unbalance/unbalance.txt").astype("float32")
        labels = np.loadtxt(f"/home/sy/PECANN-DPC/data/unbalance/unbalance.gt").flatten()  
    elif dataset == "s2":
        data = np.loadtxt(f"/home/sy/PECANN-DPC/data/s_datasets/s2.txt").astype("float32")
        labels = np.loadtxt(f"/home/sy/PECANN-DPC/data/s_datasets/s2.gt").flatten() 
    else:
        data = np.load(f"/home/sy/embeddings/{dataset}/{dataset}.npy").astype("float32")
        labels = np.loadtxt(f"/home/sy/embeddings/{dataset}/{dataset}.gt").flatten()

    print(dataset, "data loaded")

    if dataset == "unbalance":
        eps_values = np.arange(5000, 21000, 1000)
        min_samples_values = np.arange(1, 50, 2)
    elif dataset == "s2":
        eps_values = np.arange(40000, 72000, 2000)
        min_samples_values = np.concatenate((np.arange(1, 50, 2), np.arange(100, 150, 2)))
    elif dataset == "mnist":
        eps_values = np.arange(0.5, 9.5, 0.5)
        min_samples_values = np.concatenate((np.arange(1, 5, 1), np.arange(100, 1000, 200)))
    elif dataset == "birds":
        eps_values = np.concatenate((np.arange(20, 40, 5), np.arange(6, 14, 1)))
        min_samples_values = np.concatenate((np.arange(2000, 2200, 100), np.arange(1, 5, 1), np.arange(120, 270, 30)))
    elif dataset == "arxiv-clustering-s2s":
        eps_values = np.arange(0.32, 0.64, 0.02)
        min_samples_values = np.concatenate((np.arange(2000, 2200, 100), np.arange(1, 5, 1)))
    elif dataset == "imangenet":
        eps_values = np.arange(14, 34, 2)
        min_samples_values = np.concatenate((np.arange(2000, 2200, 100), np.arange(1, 5, 1), np.arange(500, 1300, 200)))
    elif dataset == "reddit-clustering":
        eps_values = np.arange(0.4, 0.72, 0.02)
        min_samples_values = np.concatenate((np.arange(2000, 2200, 100), np.arange(1, 5, 1), np.arange(3000, 13000, 1000)))
    else:
        print("wong dataset, ", dataset)
        eps_values = []
        min_samples_values = []

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
            # print(eps, min_samples, ari, clustering_time)
            # print(num_clusters, num_noise)

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)

    # Print or further analyze the results
    results_df.to_csv(f"/home/sy/PECANN-DPC/results/dbscan_c_{dataset}.csv")
    print("stored results, ", dataset)
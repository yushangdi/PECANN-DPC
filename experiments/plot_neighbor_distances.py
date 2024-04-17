import numpy as np
from sklearn.datasets import make_blobs
import sklearn
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import faiss


def plot_neighbor_distance(dataset, data, n_neighbors = 50, dimension = 784):

    # nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(data)
    # distances, indices = nbrs.kneighbors(data)
    index = faiss.IndexFlatL2(dimension)   # build the index
    index.add(data)                  # add vectors to the index
    distances, _ = index.search(data, n_neighbors)
    distances = np.sqrt(distances)

    # Plot the distance to the 50 nearest neighbors for each point
    plt.figure(figsize=(10, 6))
    # plt.hist(distances[:, -1], bins=50, density=True)
    # plt.ylabel('Density')
    # plt.title(f"Distribution of Distance to {n_neighbors} Nearest Neighbors")
    distances = distances[:, -1]
    distances = np.sort(distances)
    plt.plot(distances)
    plt.title(f"Sorted Distance to {n_neighbors} Nearest Neighbors")
    plt.savefig(f"fig_{dataset}")
    np.savetxt(f"distances_{dataset}.txt", distances)


dataset = "mnist"
dimension = 784
    
# dataset = "arxiv-clustering-s2s"
# dimension = 1024

data = np.load(f"/home/sy/embeddings/{dataset}/{dataset}.npy").astype("float32")
# dataset = "unbalance"
# data = np.loadtxt(f"../data/{dataset}/{dataset}.txt").astype("float32")
# dataset = "s2"
# data = np.loadtxt(f"../data/s_datasets/{dataset}.txt").astype("float32")

plot_neighbor_distance(dataset, data,  2 * dimension - 1, dimension)
# data, y = make_blobs(n_samples=20, centers=3, n_features=784, random_state=0)
# plot_neighbor_distance(dataset, data,  3)


# labels = np.loadtxt(f"/home/sy/embeddings/{dataset}/{dataset}.gt").flatten()
# # Count occurrences
# counts = Counter(labels)

# # Print the counts
# for item, count in counts.most_common():
#     print(f"{item}: {count}")
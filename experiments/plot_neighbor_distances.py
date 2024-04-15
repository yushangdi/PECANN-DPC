import numpy as np
from sklearn.cluster import DBSCAN
import sklearn
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter



def plot_neighbor_distance(dataset, data, n_neighbors = 50):

    # Fit nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(data)

    # Compute distances and indices of 50 nearest neighbors for each point
    distances, indices = nbrs.kneighbors(data)

    # Plot the distance to the 50 nearest neighbors for each point
    plt.figure(figsize=(10, 6))
    plt.hist(distances[:, -1], bins=50, density=True)
    plt.xlabel('Distance to 50 Nearest Neighbors')
    plt.ylabel('Density')
    plt.title(f"Distribution of Distance to {n_neighbors} Nearest Neighbors")
    plt.savefig(f"fig_{dataset}")


dataset = "mnist"
data = np.load(f"/home/sy/embeddings/{dataset}/{dataset}.npy").astype("float32")
# dataset = "unbalance"
# data = np.loadtxt(f"../data/{dataset}/{dataset}.txt").astype("float32")
# dataset = "s2"
# data = np.loadtxt(f"../data/s_datasets/{dataset}.txt").astype("float32")

plot_neighbor_distance(dataset, data, 50)

labels = np.loadtxt(f"/home/sy/embeddings/{dataset}/{dataset}.gt").flatten()
# Count occurrences
counts = Counter(labels)

# Print the counts
for item, count in counts.most_common():
    print(f"{item}: {count}")
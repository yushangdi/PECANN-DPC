import numpy as np
from sklearn.cluster import DBSCAN
import sklearn
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import time



def plot_neighbor_distance(data, n_neighbors = 50):

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
    plt.savefig("fig")


dataset = "birds"
data = np.load(f"/home/sy/embeddings/{dataset}/{dataset}.npy").astype("float32")

plot_neighbor_distance(data, 50)
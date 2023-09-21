import numpy as np

def compute_knn_and_distances(data, k):
    num_points = len(data)
    distances = np.zeros((num_points, num_points))
    
    # Compute pairwise squared distances
    for i in range(num_points):
        for j in range(num_points):
            distances[i, j] = np.sum((data[i] - data[j]) ** 2)
    
    knn_indices = np.argsort(distances, axis=1)[:, 0:k]  # Exclude the point itself
    knn_distances = np.sort(distances, axis=1)[:, 0:k]   # Exclude the point itself
    
    return knn_indices, knn_distances

# Sample data
data = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10],
    [6, 12],
    [7, 14],
    [8, 16],
    [9, 18],
    [10, 20]
])

k = 3
knn_indices, knn_distances = compute_knn_and_distances(data, k)

for i, (indices, distances) in enumerate(zip(knn_indices, knn_distances)):
    print(f"Point {i}:")
    print(f"  k-NN indices: {indices}")
    print(f"  Squared distances to k-NN: {distances}")

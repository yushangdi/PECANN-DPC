import numpy as np
import os
from sklearn.utils import shuffle
import sklearn


def _closest_center(data, centers):
    data = np.array(data)
    centers = np.array(centers)

    # Calculate the squared norms of data and centers
    data_norms_squared = np.sum(data**2, axis=1, keepdims=True)
    centers_norms_squared = np.sum(centers**2, axis=1, keepdims=True)

    # Calculate the dot product between data and centers
    dot_product = np.dot(data, centers.T)

    # Calculate the squared distances without computing the square root
    distances_squared = data_norms_squared - 2 * dot_product + centers_norms_squared.T

    # Find the index of the closest center for each point
    closest_centers = np.argmin(distances_squared, axis=1)

    return closest_centers


def generate_synthetic_data(num_datapoints, num_clusters, d=128, variance=0.05):
    points_per_cluster = num_datapoints // num_clusters
    cluster_centers = np.random.uniform(low=10, high=11, size=(num_clusters, d))
    data = []
    gts = []

    for i, center in enumerate(cluster_centers):
        data.append(
            np.random.multivariate_normal(
                mean=center, cov=np.eye(d) * variance, size=points_per_cluster
            )
        )
        gts += [i] * points_per_cluster

    data = np.vstack(data)
    data = data.astype("float32")
    gt = np.array(gts, dtype=int)
    shuffle(data, gt, random_state=0)

    closest_centers = _closest_center(data, cluster_centers)
    best_ari = sklearn.metrics.adjusted_rand_score(gt, closest_centers)
    print(points_per_cluster, "BEST", best_ari)
    return data, gt, best_ari


def _get_siblings(thread_id):
    with open(
        f"/sys/devices/system/cpu/cpu{thread_id}/topology/thread_siblings_list"
    ) as f:
        return [int(i) for i in f.readline().strip().split(",")]


def get_core_groups():
    all_threads_available = os.sched_getaffinity(0)
    main_thread_to_all = {}
    for i in all_threads_available:
        siblings = _get_siblings(i)
        if siblings[0] not in main_thread_to_all:
            main_thread_to_all[siblings[0]] = siblings
        for s in siblings:
            if s not in all_threads_available:
                raise ValueError(
                    f"Thread {s} not available to process, but is a thread sibling of {i} which is!"
                )
    return main_thread_to_all.values()

import numpy as np
import os
from sklearn.utils import shuffle


def generate_synthetic_data(num_datapoints, num_clusters, d=128, variance=0.05):
    points_per_cluster = num_datapoints // num_clusters
    cluster_centers = np.random.uniform(size=(num_clusters, d))
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

    return data, gt


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

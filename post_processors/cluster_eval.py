import numpy as np
from scipy import stats
from collections import Counter
import sklearn
from sklearn import metrics
import os
import json
import sys


def eval_clusters(
    labels,
    preds,
    verbose=True,
    eval_metrics=[
        "recall50",
        "precision50",
        "AMI",
        "ARI",
        "completeness",
        "homogeneity",
    ],
):
    label_counter = Counter(labels)
    pred_counter = Counter(preds)
    if verbose:
        print("groud truth", label_counter)
        print("clustering", pred_counter)

    result = {}

    if "recall50" in eval_metrics or "precision50" in eval_metrics:
        TP_count = 0
        for label, label_count in label_counter.items():
            ids = np.argwhere(labels == label)[:, 0]
            pred, pred_count = stats.mode(preds[ids], axis=None, keepdims=False)
            # print(label, pred)
            if pred_count / (label_count + pred_counter[pred] - pred_count) > 0.5:
                # print('pass', label, pred)
                TP_count += 1

        recall50 = TP_count / len(label_counter)
        precision50 = TP_count / len(pred_counter)

        if "recall50" in eval_metrics:
            result["recall50"] = recall50
        if "precision50" in eval_metrics:
            result["precision50"] = precision50

    if "AMI" in eval_metrics:
        result["AMI"] = sklearn.metrics.adjusted_mutual_info_score(labels, preds)

    if "ARI" in eval_metrics:
        result["ARI"] = sklearn.metrics.adjusted_rand_score(labels, preds)

    if "completeness" in eval_metrics:
        result["completeness"] = sklearn.metrics.completeness_score(labels, preds)

    if "homogeneity" in eval_metrics:
        result["homogeneity"] = sklearn.metrics.homogeneity_score(labels, preds)

    return result


def eval_clusters_wrapper(
    gt_path,
    found_clusters,
    verbose=True,
    eval_metrics=[
        "recall50",
        "precision50",
        "AMI",
        "ARI",
        "completeness",
        "homogeneity",
    ],
):
    
    with open(gt_path, "r") as file:
        labels = np.array([int(line.rstrip()) for line in file])

    if isinstance(found_clusters, str):
        with open(found_clusters, "r") as file:
            found_clusters = np.array([int(line.rstrip()) for line in file])

    return eval_clusters(labels, found_clusters, verbose, eval_metrics)


if __name__ == "__main__":
    assert len(sys.argv) >= 3
    gt_path = sys.argv[1]
    cluster_path = sys.argv[2]
    print(json.dumps(eval_clusters_wrapper(gt_path, cluster_path), indent=4))

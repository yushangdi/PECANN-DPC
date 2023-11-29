import faiss
import numpy as np
import time
from sklearn import metrics
import dpc_ann
from fastdp import fastdp

settings = {
    "s2": {"dependant_dist_threshold": 102873},
    "unbalance": {"dependant_dist_threshold": 30000},
}

result_file = "results/small_results.csv"

with open(result_file, "w") as f:
    f.write("Dataset & Algorithm & Details & Time & ARI\n")
    for dataset_name, dataset_path, gt_path in [
        ("s2", "data/s_datasets/s2.txt", "data/s_datasets/s2.gt"),
        ("unbalance", "data/unbalance/unbalance.txt", "data/unbalance/unbalance.gt"),
    ]:
        data = np.loadtxt(dataset_path)
        gt = np.loadtxt(gt_path)
        num_gt_clusters = len(set(gt))
        d = data.shape[1]

        start = time.time()
        preds = fastdp(
            data,
            num_gt_clusters,
            distance="l2",
            num_neighbors=20,
            window=50,
            nndes_start=0.2,
            maxiter=30,
            endcond=0.001,
            dtype="vec",
        )[0]
        f.write(
            f"{dataset_name} & \\algname{{fastdp}} & NA & {time.time() - start:.3f} & {metrics.adjusted_rand_score(gt, preds):.3f}\\\\\n"
        )

        data = data.astype("float32")

        kmeans = faiss.Kmeans(d, num_gt_clusters, niter=20, nredo=1, verbose=False)
        start = time.time()
        kmeans.train(data)
        preds = kmeans.index.search(data, 1)[1].flatten()
        f.write(
            f"{dataset_name} & \\algname{{$k$-means}} & nreo=1 & {time.time() - start:.3f} & {metrics.adjusted_rand_score(gt, preds):.3f}\\\\\n"
        )

        kmeans = faiss.Kmeans(d, num_gt_clusters, niter=20, nredo=50, verbose=False)
        start = time.time()
        kmeans.train(data)
        preds = kmeans.index.search(data, 1)[1].flatten()
        f.write(
            f"{dataset_name} & \\algname{{$k$-means}} & nredo=50 & {time.time() - start:.3f} & {metrics.adjusted_rand_score(gt, preds):.3f}\\\\\n"
        )

        start = time.time()
        data = np.pad(data, [(0, 0), (0, 6)])
        preds = dpc_ann.dpc_numpy(
            data=data,
            center_finder=dpc_ann.ProductCenterFinder(num_gt_clusters),
        ).clusters
        f.write(
            f"{dataset_name} & \\framework & product center finder & {time.time() - start:.3f} & {metrics.adjusted_rand_score(gt, preds):.3f}\\\\\n"
        )

        start = time.time()
        preds = dpc_ann.dpc_numpy(
            data=data,
            center_finder=dpc_ann.ThresholdCenterFinder(**settings[dataset_name]),
        ).clusters
        f.write(
            f"{dataset_name} & \\framework & threshold center finder & {time.time() - start:.3f} & {metrics.adjusted_rand_score(gt, preds):.3f}\\\\\n"
        )

        start = time.time()
        preds = dpc_ann.dpc_numpy(
            data=data,
            method="BruteForce",
            center_finder=dpc_ann.ThresholdCenterFinder(**settings[dataset_name]),
        ).clusters
        f.write(
            f"{dataset_name} & \\algname{{BruteForce}} & threshold center finder & {time.time() - start:.3f} & {metrics.adjusted_rand_score(gt, preds):.3f}\\\\\n"
        )

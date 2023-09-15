import dpc_ann
import numpy as np

data = np.load("data/imagenet/imagenet.npy")

method = "BruteForce"

dpc_ann.dpc_numpy(
    data=data,
    decision_graph_path=f"results/imagenet/imagenet_{method}.dg",
    graph_type=method,
    output_path=f"results/imagenet/imagenet_{method}.cluster"
)

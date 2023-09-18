import dpc_ann
import numpy as np
from pathlib import Path
import os

# Change to DPC-ANN folder
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)

method = "BruteForce"
dataset = "mnist"

data = np.load(f"data/{dataset}/{dataset}.npy")


dpc_ann.dpc_numpy(
    data=data,
    decision_graph_path=f"results/{dataset}/{dataset}_{method}.dg",
    graph_type=method,
    output_path=f"results/{dataset}/{dataset}_{method}.cluster",
)

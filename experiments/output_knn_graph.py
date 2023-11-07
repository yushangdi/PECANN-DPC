#!/usr/bin/env python3

import itertools
import os
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import multiprocessing
import argparse
from sklearn import datasets

import dpc_ann

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

# data = np.load(f"data/{dataset_folder}/{dataset}.npy").astype("float32")
# digits = datasets.load_digits()
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1)).astype("float32")
# knn_graph_path = "results/knn_graphs/digits.graph.txt"

data = np.load("data/mnist/mnist.npy")
print("data loaded")
ks = [10, 50]
for k in ks:
  knn_graph_path = "results/knn_graphs/mnist_k%s.graph.txt" % k
  dpc_ann.dpc_numpy(
              graph_type="BruteForce",
              knn_graph_path=knn_graph_path,
              data=data,
              K=k,
              center_finder=dpc_ann.ProductCenterFinder(num_clusters=1),
          )


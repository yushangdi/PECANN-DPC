import faiss
import numpy as np
from post_processors.cluster_eval import eval_clusters
import sys
import os
from pathlib import Path

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))


x = np.load("data/imagenet/imagenet.npy")
ncentroids = 1000


niter = 20
verbose = True
d = x.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
kmeans.train(x)

_, clusters = kmeans.index.search(x, 1)
clusters = clusters.flatten()

ground_truth = np.loadtxt("data/imagenet/imagenet.gt", dtype=int)

eval_clusters(ground_truth, clusters, verbose=False)

# Result for imagenet:
# {'recall50': 0.692,
#  'precision50': 0.692,
#  'AMI': 0.8810204445757013,
#  'ARI': 0.6431451638897032,
#  'completeness': 0.895803375866884,
#  'homogeneity': 0.8815404215715357}

# Result for mnist, 0.4 recall and precision and ARI

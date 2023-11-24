import os
from pathlib import Path
import sys

abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from test_kmeans import run_kmeans_experiment

for dataset, num_clusters in [
    ("mnist", 10),
    ("imagenet", 1000),
    ("arxiv-clustering-s2s", 180),
    ("reddit-clustering", 50),
    ("birds", 525),
]:
    run_kmeans_experiment(dataset, prefix="kmeans_pareto")

import os
from pathlib import Path
import sys

abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from test_dpc_ann import run_dpc_ann_configurations

Ks = [8, 16]
for dataset, num_clusters in [
    ("mnist", 10),
    ("arxiv-clustering-s2s", 180),
    ("reddit-clustering", 50),
    ("birds", 525),
    ("imagenet", 1000),
]:
    run_dpc_ann_configurations(
        dataset,
        timeout_s=1000000,
        num_clusters=num_clusters,
        graph_types=["BruteForce"],
        compare_against_bf=False,
        density_methods=["kth", "sum-exp"],
        Ks=Ks,
        results_file_prefix=f"bruteforce_{dataset}",
    )

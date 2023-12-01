import os
from pathlib import Path
import sys

abspath = Path(__file__).resolve().parent.parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from test_dpc_ann import run_dpc_ann_configurations

for dataset, num_clusters in [
    ("mnist", 10),
    ("imagenet", 1000),
    ("arxiv-clustering-s2s", 180),
    ("reddit-clustering", 50),
    ("birds", 525),
]:
    run_dpc_ann_configurations(
        dataset,
        timeout_s=400,
        num_clusters=num_clusters,
        graph_types=["Vamana"],
        compare_against_bf=True,
        density_methods=["kth"],
        Ks=[16],
        results_file_prefix=f"vamana_pareto_{dataset}",
    )

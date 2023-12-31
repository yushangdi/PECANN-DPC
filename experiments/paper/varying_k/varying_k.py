import os
from pathlib import Path
import sys

abspath = Path(__file__).resolve().parent.parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from test_dpc_ann import run_dpc_ann_configurations

Ks = [4, 8, 16, 32, 64]
for dataset, num_clusters, param_value in [
    ("mnist", 10, 32),
    ("imagenet", 1000, 128),
    ("arxiv-clustering-s2s", 180, 64),
    ("reddit-clustering", 50, 64),
    ("birds", 525, 32),
]:
    run_dpc_ann_configurations(
        dataset,
        timeout_s=2000,
        num_clusters=num_clusters,
        graph_types=["Vamana"],
        search_range=[param_value],
        compare_against_bf=False,
        density_methods=["kth", "normalized", "exp-sum", "sum-exp", "sum"],
        Ks=Ks,
        results_file_prefix=f"varying_k_{dataset}",
    )

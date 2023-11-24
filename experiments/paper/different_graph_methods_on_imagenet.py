import os
from pathlib import Path
import sys

abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from test_dpc_ann import run_dpc_ann_configurations

run_dpc_ann_configurations(
    "imagenet",
    timeout_s=300,
    num_clusters=1000,
    compare_against_bf=True,
    density_methods=["kth"],
    graph_types=["HCNNG", "pyNNDescent"],  # Can add Vamana from the other experiments
    Ks=[16],
    results_file_prefix=f"imagenet_different_methods",
)

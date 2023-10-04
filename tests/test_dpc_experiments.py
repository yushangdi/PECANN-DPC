import os
from pathlib import Path
import sys
import pandas as pd

# Change to experiments folder to import basic_experiments
abspath = Path(__file__).resolve().parent.parent / "experiments"
os.chdir(abspath)
sys.path.append(str(abspath))

from experiments.basic_experiments import run_basic_experiments
from experiments.test_dpc_ann import run_dpc_ann_configurations


# Just test that we can run the experiments at all with the new and old framework
def test_basic_experiments():
    run_basic_experiments(new_framework=True)
    run_basic_experiments(new_framework=False)


# Test both old and new framework gets good ARI, and ARI is pretty close
# Note that the ARI is not exactly the same because of the neighbor optimization, but
# if that is removed it is indeed exactly the same.
def test_mnist_ari():
    results = pd.read_csv(
        run_dpc_ann_configurations(
            dataset="mnist",
            timeout_s=50,
            num_clusters=10,
            graph_types=["Vamana"],
            search_range=[32],
            compare_against_gt=True,
            run_new_dpc_framework=True,
            run_old_dpc_framework=True,
        )
    )
    assert (results["ARI"][results["comparison"] == "ground truth"] > 0.3).all()
    assert (results["ARI"][results["comparison"] == "brute force"] > 0.95).all()

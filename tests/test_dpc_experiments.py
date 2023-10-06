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


# TODO: There is an extremely weird bug where if we reverse the order of these
# two tests, test_mnist_ari does not use multiple threads (but this is not the
# case for test_basic_experiments in the current order). This might be a strange
# parlay bug, I'm not sure, it might depend on how python isolates tests.


# Test both old and new framework gets good ARI, and ARI is pretty close
# Note that the ARI is not exactly the same between the old code and the new
# framework, because of the neighbor optimization, but if that is removed it
# is indeed exactly the same.
def test_mnist_ari():
    results = pd.read_csv(
        run_dpc_ann_configurations(
            dataset="mnist",
            timeout_s=100,
            num_clusters=10,
            graph_types=["Vamana"],
            search_range=[32],
            compare_against_gt=True,
        )
    )
    assert len(results) == 2
    assert (results["ARI"][results["comparison"] == "ground truth"] > 0.3).all()
    assert (results["ARI"][results["comparison"] == "brute force"] > 0.95).all()


# Just test that we can run the basic experiments at all
def test_basic_experiments():
    run_basic_experiments()

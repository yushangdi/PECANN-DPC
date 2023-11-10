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
    assert len(results) == 4
    assert (results["ARI"][results["comparison"] == "ground truth"] > 0.3).all()

    brute_force_results = results[results["method"].str.startswith("BruteForce")]
    vamana_results = results[results["method"].str.startswith("Vamana")]
    assert len(brute_force_results) == 2
    assert len(vamana_results) == 2

    assert (
        brute_force_results["ARI"][results["comparison"] == "brute force"] == 1.0
    ).all()
    assert (vamana_results["ARI"][results["comparison"] == "brute force"] > 0.95).all()


# Just test that we can run the basic experiments at all
def test_basic_experiments():
    run_basic_experiments()

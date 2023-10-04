import os
from pathlib import Path
import sys

# Change to experiments folder to import basic_experiments
abspath = Path(__file__).resolve().parent.parent / "experiments"
os.chdir(abspath)
sys.path.append(str(abspath))

from experiments.basic_experiments import run_basic_experiments


# Just test that we can run the experiments at all with the new and old framework
def test_basic_experiments_run():
    run_basic_experiments(new_framework=True)
    run_basic_experiments(new_framework=False)

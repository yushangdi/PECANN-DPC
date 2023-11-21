import os
from pathlib import Path
import sys
from paper_utils import get_core_groups
import multiprocessing

abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))


def run_all_datasets_restrict_threads(current_threads):
    num_threads = len(current_threads)
    os.sched_setaffinity(0, current_threads)
    os.environ["PARLAY_NUM_THREADS"] = f"{num_threads}"

    from test_dpc_ann import run_dpc_ann_configurations

    for dataset, num_clusters, param_value in [
        ("mnist", 10, 32),
        ("imagenet", 1000, 128),
        ("arxiv-clustering-s2s", 180, 64),
        ("reddit-clustering", 50, 64),
        ("birds", 525, 32),
    ]:
        run_dpc_ann_configurations(
            dataset,
            timeout_s=10000,
            num_clusters=num_clusters,
            graph_types=["Vamana"],
            search_range=[param_value],
            compare_against_bf=False,
            density_methods=["kth"],
            Ks=[16],
            results_file_prefix=f"restricted_{dataset}_to_{num_threads}_threads",
        )


def run_experiment():
    current_threads = []
    core_groups = get_core_groups()

    def flatten(l):
        return [item for sublist in l for item in sublist]

    threads = flatten(core_groups)

    print(threads)
    for thread in threads:
        current_threads.append(thread)
        if len(current_threads) in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            p = multiprocessing.Process(
                target=run_all_datasets_restrict_threads, args=(current_threads,)
            )
            p.start()
            p.join()


if __name__ == "__main__":
    run_experiment()

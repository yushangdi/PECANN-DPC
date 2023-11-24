import os
from pathlib import Path
import sys
import multiprocessing


abspath = Path(__file__).resolve().parent.parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))


def _get_siblings(thread_id):
    with open(
        f"/sys/devices/system/cpu/cpu{thread_id}/topology/thread_siblings_list"
    ) as f:
        return [int(i) for i in f.readline().strip().split(",")]


def get_core_groups():
    all_threads_available = os.sched_getaffinity(0)
    main_thread_to_all = {}
    for i in all_threads_available:
        siblings = _get_siblings(i)
        if siblings[0] not in main_thread_to_all:
            main_thread_to_all[siblings[0]] = siblings
        for s in siblings:
            if s not in all_threads_available:
                raise ValueError(
                    f"Thread {s} not available to process, but is a thread sibling of {i} which is!"
                )
    return main_thread_to_all.values()


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
            results_file_prefix=f"restricted_{dataset}_to_{num_threads}_cores",
        )


def run_experiment():
    current_threads = []
    core_groups = get_core_groups()

    def flatten(l):
        return [item for sublist in l for item in sublist]

    threads = flatten(core_groups)

    print(threads)
    for core_group in core_groups:
        current_threads.append(core_group[0])
        # Neccesary to rerun with just len(current_threads) in [60] and
        # numactl -i all to get the best performance
        if len(current_threads) in [1, 2, 4, 8, 15, 30]:
            p = multiprocessing.Process(
                target=run_all_datasets_restrict_threads, args=(current_threads,)
            )
            p.start()
            p.join()


if __name__ == "__main__":
    run_experiment()

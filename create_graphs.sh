#!/bin/bash

python3 experiments/paper/dataset_size_scaling/plot_dataset_size_scaling.py \
    /data/scratch/jae/dpc_ann_results/cluster_analysis_dataset_scaling.csv

python3 experiments/paper/pareto/plot_pareto.py /data/scratch/jae/dpc_ann_results

python3 experiments/paper/varying_k/plot_varying_k.py /data/scratch/jae/dpc_ann_results

python3 experiments/paper/thread_scaling/plot_thread_scaling.py \
    /data/scratch/jae/dpc_ann_results/thread_scaling --threads

python3 experiments/paper/thread_scaling/plot_thread_scaling.py \
    /data/scratch/jae/dpc_ann_results/core_scaling --no-threads

python3 experiments/paper/varying_num_clusters/plot_varying_num_clusters.py  \
   /data/scratch/jae/dpc_ann_results/cluster_analysis_varying_num_clusters.csv
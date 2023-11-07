# DPC-ANN

![example workflow](https://github.com/yushangdi/DPC-ANN/actions/workflows/cmake-single-platform.yml/badge.svg)

```bash
pip3 install nanobind
sudo apt-get install libboost-all-dev
mkdir build
cd build
cmake ..
make
cd ../
pip3 install .
```

Run directly from commandline:
```bash
./build/dpc_ann_exe  --query_file ./data/gaussian_example/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000.dg 

./build/dpc_framework_exe  --query_file ./data/gaussian_example/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000.dg --output_file ./results/gaussian_4_1000.cluster --dist_cutoff 8.36 
```

Test pip3 installation
```bash
python3 tests/test_python_install.py
```

Test dpc frameworks
```bash
./build/dpc_tests
```

## PyNNDescent

`max_degree` is the k used to build the graph. We look at the 2-hop neighbors of each node in each round, and retain the k-th closeset neighbors.

`Lbuild` ->cluster_size,  the leaf size of the cluster trees.
`num_clusters`: only used for pyNNDescent. the number of cluster trees to use when initializing the graph. roughly linear with the graph constructin time.


`alpha`: prune parameter, similar to Vamana.

-Lbuild 100 -num_clusters 10

Only graph construction is different. All other parts are the same.

<!-- (v, k, R, beamSize, beamSizeQ, alpha, delta, qpts, groundTruth, res_file, graph_built, D) -->

```bash
 ./doubling_dpc --query_file ./data/unbalance.txt --decision_graph_path ./results/unbalance.dg --output_file ./results/unbalance.cluster --dist_cutoff 30000 --graph_type p --Lbuild 200 --num_clusters 1
 ```

 ## NCHHG

`max_degree` is the MST degree used.

# Old Commands 

Change line 3 in Makefile to the location of boost library.
```bash
sudo apt-get install libboost-program-options-dev
git submodule init
git submodule update
cd ParlayANN
git submodule init
git submodule update
cd ../
make
```

Write info for plotting decision graph.
```bash
mkdir results
./doubling_dpc --query_file ./data/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000.dg 
```

plot decision graph:
```bash
python3 post_processors/plot_decision_graph.py results/gaussian_4_1000.dg 4 
python3 post_processors/plot_decision_graph.py results/gaussian_4_1000_bruteforce.dg 4 2
python3 post_processors/plot_decision_graph.py results/gaussian_4_10000_bruteforce.dg 4 2
python3 post_processors/plot_decision_graph.py results/gaussian_4_10000_128.dg 4
python3 post_processors/plot_decision_graph.py results/gaussian_4_10000_128_bruteforce.dg 4 2
python3 post_processors/plot_decision_graph.py results/gaussian_4_1000_128.dg 4 1000_128
python3 post_processors/plot_decision_graph.py results/gaussian_4_1000_128_bruteforce.dg 4 1000_128_bruteforce
python3 post_processors/plot_decision_graph.py results/gaussian_4_10000_128.dg 4 10000_128
python3 post_processors/plot_decision_graph.py results/gaussian_4_10000_128_priority.dg 4 10000_128_priority
```

Write cluster result
```bash
./doubling_dpc --query_file ./data/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000.dg --dist_cutoff 95 --output_file ./results/gaussian_4_1000.cluster
./doubling_dpc --query_file ./data/gaussian_4_10000.data --decision_graph_path ./results/gaussian_4_10000.dg --dist_cutoff 726 --output_file ./results/gaussian_4_10000.cluster --Lbuild 6
./doubling_dpc --query_file ./data/gaussian_4_10000_128.data --decision_graph_path ./results/gaussian_4_10000_128.dg --dist_cutoff 250 --output_file ./results/gaussian_4_10000_128.cluster --Lbuild 15
./doubling_dpc --query_file ./data/gaussian_4_1000_128.data --decision_graph_path ./results/gaussian_4_1000_128.dg --dist_cutoff 94496 --output_file ./results/gaussian_4_1000_128.cluster --Lbuild 8
 ```

Evaluate clustering
```bash
python post_processors/cluster_eval.py data/gaussian_example/gaussian_4_1000.gt results/gaussian_4_1000.cluster 
python post_processors/cluster_eval.py data/gaussian_4_10000.gt results/gaussian_4_10000.cluster 
python post_processors/cluster_eval.py data/gaussian_4_10000.gt results/gaussian_4_10000_priority.cluster 
python post_processors/cluster_eval.py data/gaussian_4_1000_128.gt results/gaussian_4_1000_128.cluster 
python post_processors/cluster_eval.py data/gaussian_4_10000_128.gt results/gaussian_4_10000_128_priority.cluster 
python post_processors/cluster_eval.py data/gaussian_4_10000_128.gt results/gaussian_4_10000_128.cluster 
```


Running bruteforce exact method:
```bash
./doubling_dpc --query_file ./data/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000_bruteforce.dg --output_file ./results/gaussian_4_1000_bruteforce.cluster --dist_cutoff 95 --bruteforce true
./doubling_dpc --query_file ./data/gaussian_4_10000_128.data --decision_graph_path ./results/gaussian_4_10000_128_bruteforce.dg --output_file ./results/gaussian_4_10000_128_bruteforce.cluster --dist_cutoff 250 --bruteforce true
./doubling_dpc --query_file ./data/gaussian_4_1000_128.data --decision_graph_path ./results/gaussian_4_1000_128_bruteforce.dg --output_file ./results/gaussian_4_1000_bruteforce.cluster --dist_cutoff 94496 --bruteforce true
```


Get S1, S2, and Unbalanced datasets and groundtruth

```bash
# source: http://cs.joensuu.fi/sipu/datasets/
wget http://cs.joensuu.fi/sipu/datasets/s2.txt
wget http://cs.joensuu.fi/sipu/datasets/s3.txt
wget http://cs.joensuu.fi/sipu/datasets/unbalance.txt
wget http://cs.joensuu.fi/sipu/datasets/s-originals.zip
wget http://cs.joensuu.fi/sipu/datasets/unbalance-gt-pa.zip
```

Get KDD, Facial, no groundtruth

```bash
wget http://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt
wget https://archive.ics.uci.edu/static/public/317/grammatical+facial+expressions.zip
unzip grammatical+facial+expressions.zip -d faical
```

S2
```bash
# bruteforce
./doubling_dpc --query_file ./data/s_dataset/s2.txt --decision_graph_path ./results/s2_bruteforce.dg --output_file ./results/s2_bruteforce.cluster --dist_cutoff 102873 --bruteforce true
python3 post_processors/plot_decision_graph.py results/s2_bruteforce.dg 15 s2_bruteforce
python post_processors/cluster_eval.py ./data/s_dataset/s2-label.gt results/s2_bruteforce.cluster 
python data_processors/plot.py data/s_dataset/s2.txt results/s2_bruteforce.cluster results/s2.png 0 1

# ANN method
./doubling_dpc --query_file ./data/s_dataset/s2.txt --decision_graph_path ./results/s2.dg --output_file ./results/s2.cluster --dist_cutoff 102873
python3 post_processors/plot_decision_graph.py results/s2.dg 15 s2
python3 post_processors/cluster_eval.py ./data/s_dataset/s2-label.gt results/s2.cluster 
python3 post_processors/cluster_eval.py results/s2_bruteforce.cluster results/s2.cluster 
```



S3
```bash
# bruteforce
./doubling_dpc --query_file ./data/s_dataset/s3.txt --decision_graph_path ./results/s3_bruteforce.dg --output_file ./results/s3_bruteforce.cluster --dist_cutoff 102873 --bruteforce true
python3 post_processors/plot_decision_graph.py results/s3_bruteforce.dg 15 s3_bruteforce
python post_processors/cluster_eval.py ./data/s_dataset/s3-label.gt results/s3_bruteforce.cluster 
python data_processors/plot.py data/s_dataset/s3.txt results/s3_bruteforce.cluster results/s3.png 0 1

# ANN method
./doubling_dpc --query_file ./data/s_dataset/s3.txt --decision_graph_path ./results/s3.dg --output_file ./results/s3.cluster --dist_cutoff 102873
python3 post_processors/plot_decision_graph.py results/s3.dg 15 s3
python3 post_processors/cluster_eval.py ./data/s_dataset/s3-label.gt results/s3.cluster 
python3 post_processors/cluster_eval.py results/s3_bruteforce.cluster results/s3.cluster 
```

Unbalanced
```bash
# bruteforce
./doubling_dpc --query_file ./data/unbalance.txt --decision_graph_path ./results/unbalance_bruteforce.dg --output_file ./results/unbalance_bruteforce.cluster --dist_cutoff 30000 --bruteforce true
python3 post_processors/plot_decision_graph.py results/unbalance_bruteforce.dg 8 unbalance_bruteforce
python post_processors/cluster_eval.py ./data/unbalance.gt results/unbalance_bruteforce.cluster 
python data_processors/plot.py data/unbalance.txt results/unbalance_bruteforce.cluster results/unbalance.png 0 1

# ANN method
./doubling_dpc --query_file ./data/unbalance.txt --decision_graph_path ./results/unbalance.dg --output_file ./results/unbalance.cluster --dist_cutoff 30000 
python3 post_processors/plot_decision_graph.py results/unbalance.dg 8 unbalance
python3 post_processors/cluster_eval.py ./data/unbalance.gt results/unbalance.cluster 
python3 post_processors/cluster_eval.py results/unbalance_bruteforce.cluster results/unbalance.cluster 
python data_processors/plot.py data/unbalance.txt results/unbalance.cluster results/unbalance.png 0 1
```

Facial
```bash
# bruteforce
./doubling_dpc --query_file ./data/unbalance.txt --decision_graph_path ./results/unbalance_bruteforce.dg --output_file ./results/unbalance_bruteforce.cluster --dist_cutoff 30000 --bruteforce true
python3 post_processors/plot_decision_graph.py results/unbalance_bruteforce.dg 8 unbalance_bruteforce
python post_processors/cluster_eval.py ./data/unbalance.gt results/unbalance_bruteforce.cluster 
python data_processors/plot.py data/unbalance.txt results/unbalance_bruteforce.cluster results/unbalance.png 0 1

# ANN method
./doubling_dpc --query_file ./data/unbalance.txt --decision_graph_path ./results/unbalance.dg --output_file ./results/unbalance.cluster --dist_cutoff 30000 
python3 post_processors/plot_decision_graph.py results/unbalance.dg 8 unbalance
python3 post_processors/cluster_eval.py ./data/unbalance.gt results/unbalance.cluster 
python3 post_processors/cluster_eval.py results/unbalance_bruteforce.cluster results/unbalance.cluster 
python data_processors/plot.py data/unbalance.txt results/unbalance.cluster results/unbalance.png 0 1
```


MNIST
```bash
# bruteforce
./doubling_dpc --query_file ./data/mnist.txt --decision_graph_path ./results/mnist_bruteforce.dg --output_file ./results/mnist_bruteforce.cluster --dist_cutoff 8 --bruteforce true
python3 post_processors/plot_decision_graph.py results/mnist_bruteforce.dg 10 mnist_bruteforce
python post_processors/cluster_eval.py ./data/mnist.gt results/mnist_bruteforce.cluster 
python data_processors/plot.py data/mnist.txt results/mnist_bruteforce.cluster results/mnist.png 0 1

```


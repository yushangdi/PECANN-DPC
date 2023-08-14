# DPC-ANN


Change line 3 in Makefile to the location of boost library.
```bash
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
python post_processors/cluster_eval.py data/gaussian_4_1000.gt results/gaussian_4_1000.cluster 
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

Get KDD, no groundtruth

```bash
wget http://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt
```

S2
```bash
./doubling_dpc --query_file ./data/s_dataset/s2.txt --decision_graph_path ./results/s2_bruteforce.dg --output_file ./results/s2_bruteforce.cluster --dist_cutoff 102873 --bruteforce true
python3 post_processors/plot_decision_graph.py results/s2_bruteforce.dg 15 s2_bruteforce
python post_processors/cluster_eval.py ./data/s_dataset/s2-label.gt results/s2_bruteforce.cluster 
python data_processors/plot.py data/s_dataset/s2.txt results/s2_bruteforce.cluster results/s2.png 0 1


./doubling_dpc --query_file ./data/s_dataset/s2.txt --decision_graph_path ./results/s2.dg --output_file ./results/s2.cluster --dist_cutoff 102873
python3 post_processors/plot_decision_graph.py results/s2.dg 15 s2
python3 post_processors/cluster_eval.py ./data/s_dataset/s2-label.gt results/s2.cluster 
python3 post_processors/cluster_eval.py results/s2_bruteforce.cluster results/s2.cluster 

```
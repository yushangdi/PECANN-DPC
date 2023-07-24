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
python3 post_processors/plot_decision_graph.py results/gaussian_4_1000.dg 
```

Write cluster result
```bash
./doubling_dpc --query_file ./data/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000.dg --dist_cutoff 95 --output_file ./results/gaussian_4_1000.cluster
 ```

Evaluate clustering
```bash
 python post_processors/cluster_eval.py data/gaussian_4_1000.gt results/gaussian_4_1000.cluster 
```


Running bruteforce exact method:
```bash
./doubling_dpc --query_file ./data/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000_bruteforce.dg --output_file ./results/gaussian_4_1000_bruteforce.cluster --dist_cutoff 95 --bruteforce true
using brute force
```
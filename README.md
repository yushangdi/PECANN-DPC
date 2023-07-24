# DPC-ANN


Change line 3 in Makefile to the location of boost library.


```bash
make
mkdir results
./doubling_dpc --query_file ./data/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000.dg 
```

plot decision graph:

```bash
python3 post_processors/plot_decision_graph.py results/gaussian_4_1000.dg 
```
# DPC-ANN

![example workflow](https://github.com/yushangdi/DPC-ANN/actions/workflows/build-package-and-run-tests.yml/badge.svg)

```bash
pip3 install nanobind
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

MNIST
```bash
# bruteforce
./doubling_dpc --query_file ./data/mnist.txt --decision_graph_path ./results/mnist_bruteforce.dg --output_file ./results/mnist_bruteforce.cluster --dist_cutoff 8 --bruteforce true
python3 post_processors/plot_decision_graph.py results/mnist_bruteforce.dg 10 mnist_bruteforce
python post_processors/cluster_eval.py ./data/mnist.gt results/mnist_bruteforce.cluster 
python data_processors/plot.py data/mnist.txt results/mnist_bruteforce.cluster results/mnist.png 0 1

```


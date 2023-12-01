# PECANN: Parallel Efficient Clustering with Approximate Nearest Neighbors

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
./build/dpc_framework_exe  --query_file ./data/gaussian_example/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000.dg --output_file ./results/gaussian_4_1000.cluster --dist_cutoff 8.36 
```

Test pip3 installation
```bash
python3 tests/test_python_install.py
```

```
@software{Yu_PECANN_Parallel_Efficient_2023,
author = {Yu, Shangdi and Engels, Joshua  and Huang, Yihao and Shun, Julian },
month = dec,
title = {{PECANN: Parallel Efficient Clustering with Graph-Based Approximate Nearest Neighbor Search}},
url = {https://github.com/yushangdi/PECANN-DPC/},
version = {1.0.0},
year = {2023}
}
```

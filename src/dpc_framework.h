#pragma once

#include "IO.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

#include "ParlayANN/algorithms/HCNNG/hcnng_index.h"
#include "ParlayANN/algorithms/pyNNDescent/pynn_index.h"
#include "ParlayANN/algorithms/utils/NSGDist.h"
#include "ParlayANN/algorithms/utils/beamSearch.h"
#include "ParlayANN/algorithms/utils/parse_files.h"
#include "ParlayANN/algorithms/vamana/neighbors.h"

#include "ann_utils.h"
#include "bruteforce.h"
#include "union_find.h"
#include "utils.h"

namespace DPC {

// This struct does not own the passed in pointer
struct DatasetKnn {
  size_t num_data_;
  size_t data_dim_;
  size_t aligned_dim_;
  const float *data_;
  const Distance *D_;
  const std::pair<int, double> *knn_;
  const int k_;

  DatasetKnn(const RawDataset &raw_data, const Distance *D, const int k,
             const std::vector<std::pair<int, double>> &knn)
      : num_data_(raw_data.num_data), data_dim_(raw_data.data_dim),
        aligned_dim_(raw_data.aligned_dim), data_(raw_data.data), D_(D),
        knn_(knn.data()), k_(k) {}
};

// Construct graph for `data`. Parameters:
//   raw_data: raw input data points corresponding to data.
//   D: distance computation method
//   L: the buffer size parameter
template <class T>
parlay::sequence<Tvec_point<T> *>
construct_graph(const RawDataset &raw_data, const unsigned L, const float alpha,
                const int max_degree, const Distance *D,
                const GraphType graph_type);

// Compute the k-nearest neighbors of points in `graph` using graph-based
// method. Parameters:
//   graph: input data points. They should have neighbors computed already.
//   raw_data: raw input data points corresponding to data.
//   D: distance computation method
//   K: the k in knn
//   L: the buffer size parameter
template <class T>
std::vector<std::pair<int, double>>
compute_knn(parlay::sequence<Tvec_point<T> *> &graph,
            const RawDataset &raw_data, const unsigned K, const unsigned L,
            const Distance *D);

// Compute the k-nearest neighbors of points in `raw_data` using bruteforce
// method. Parameters:
//   raw_data: raw input data points corresponding to data.
//   D: distance computation method
//   K: the k in knn
template <class T>
std::vector<std::pair<int, double>>
compute_knn_bruteforce(const RawDataset &raw_data, const unsigned K,
                       const Distance *D);

// Compute the dependet point of points in `graph` with densities above
// `density_cutoff` using graph-based method. Parameters:
//   graph: input data points. They should have neighbors computed already.
//   raw_data: raw input data points corresponding to data.
//   densities: densities of data pints.
//   density_cutoff: points with densities below this threshold are considered
//   noise points, and their dependent points are not computed.
//   D: distance computation method
//   L: the buffer size parameter
//   round_limit: the round limit parameter
template <class T>
std::vector<std::pair<int, double>>
compute_dep_ptr(parlay::sequence<Tvec_point<T> *> &graph,
                const RawDataset &raw_data, const std::vector<T> &densities,
                const Distance *D, unsigned L, int round_limit = -1);

// Compute the dependet point of `raw_data` with densities above
// `density_cutoff` using bruteforce method. Parameters:
//   raw_data: input data points.
//   densities: densities of data pints.
//   density_cutoff: points with densities below this threshold are considered
//   noise points, and their dependent points are not computed.
//   D: distance computation method
template <class T>
std::vector<std::pair<int, double>>
compute_dep_ptr_bruteforce(const RawDataset &raw_data,
                           const std::vector<T> &densities,
                           const float density_cutoff, Distance *D);

// Base class for other DPCComputers.
class DPCComputer {
protected:
  const std::pair<int, double> *knn_;
  const Distance *D_;
  size_t num_data_;
  size_t data_dim_;
  size_t aligned_dim_;
  const float *data_;
  const int k_;

  DPCComputer(const DatasetKnn &data_knn)
      : knn_(data_knn.knn_), D_(data_knn.D_), num_data_(data_knn.num_data_),
        data_dim_(data_knn.data_dim_), aligned_dim_(data_knn.aligned_dim_),
        data_(data_knn.data_), k_(data_knn.k_) {}

  virtual ~DPCComputer() {} // Virtual destructor
};

template <typename T> class DensityComputer : public DPCComputer {
public:
  // Here we're passing the necessary arguments to the base class constructor
  DensityComputer(const DatasetKnn &data_knn) : DPCComputer(data_knn) {}

  // Return the density.
  virtual std::vector<double>
  operator()(parlay::sequence<Tvec_point<T> *> &graph);

  // Reweight the density of each point in $v$ based on knn.
  virtual std::vector<double> reweight_density(const std::vector<T> &densities);
};

template <typename T> class CenterFinder : public DPCComputer {
public:
  CenterFinder(const DatasetKnn &data_knn) : DPCComputer(data_knn) {}

  virtual ~CenterFinder() {}

  virtual std::set<int>
  operator()(std::vector<T> &densities, std::vector<T> &re_weighted_densities,
             std::vector<std::pair<uint32_t, double>> &dep_ptrs) = 0;
};

// Now, apply the same constructor pattern for the other classes:

template <typename T> class NoiseFinder : public DPCComputer {
public:
  NoiseFinder(const DatasetKnn &data_knn) : DPCComputer(data_knn) {}

  virtual ~NoiseFinder() {}

  virtual std::set<int> operator()(std::vector<T> &densities,
                                   std::vector<T> &re_weighted_densities) = 0;
};

template <typename T> class ThresholdCenterFinder : public CenterFinder<T> {
private:
  double delta_threshold_;
  double density_threshold_;

public:
  ThresholdCenterFinder(const DatasetKnn &data_knn, double delta_threshold,
                        double density_threshold)
      : DPCComputer(data_knn), delta_threshold_(delta_threshold),
        density_threshold_(density_threshold) {}

  virtual ~ThresholdCenterFinder() {}

  virtual std::set<int>
  operator()(std::vector<T> &densities, std::vector<T> &re_weighted_densities,
             std::vector<std::pair<uint32_t, double>> &dep_ptrs) = 0;
};

template <typename T> class ClusterAssigner : public DPCComputer {
public:
  ClusterAssigner(const DatasetKnn &data_knn) : DPCComputer(data_knn) {}

  virtual std::vector<int>
  operator()(std::vector<T> &densities, std::vector<T> &re_weighted_densities,
             std::vector<std::pair<uint32_t, double>> &dep_ptrs) = 0;
};

template <typename T> class ClusterMerger : public DPCComputer {
public:
  ClusterMerger(const DatasetKnn &data_knn) : DPCComputer(data_knn) {}

  virtual std::vector<int>
  operator()(std::vector<int> &current_cluster, std::vector<T> &densities,
             std::vector<T> &re_weighted_densities,
             std::vector<std::pair<uint32_t, double>> &dep_ptrs) = 0;
};

} // namespace DPC
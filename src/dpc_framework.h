#pragma once

#include "IO.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
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
  // the i*k+j's element is i's jth nearest neighbor.
  // the distance is squared distance
  const std::pair<int, double> *knn_;
  const int k_;

  DatasetKnn(const RawDataset &raw_data, const Distance *D, const int k,
             const std::vector<std::pair<int, double>> &knn)
      : num_data_(raw_data.num_data), data_dim_(raw_data.data_dim),
        aligned_dim_(raw_data.aligned_dim), data_(raw_data.data), D_(D),
        knn_(knn.data()), k_(k) {}

  // return the dependent point of i if it's in i'th knn, return nullopt
  // otherwise.
  std::optional<std::pair<int, float>>
  get_dep_ptr(int i, std::vector<double> &densities) {
    double d_i = densities[i];
    for (size_t j = 0; j < k_; ++j) {
      int id = knn_[i * k_ + j].first;
      float d_j = densities[id];
      if (d_j > d_i || (d_i == d_i && id > i)) {
        return knn_[i * k_ + j].second;
      }
    }
    return std::nullopt;
  }
};

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

  DPCComputer() {}

  void initialize(const DatasetKnn &data_knn) {
    knn_ = data_knn.knn_;
    D_ = data_knn.D_;
    num_data_ = data_knn.num_data_;
    data_dim_ = data_knn.data_dim_;
    aligned_dim_ = data_knn.aligned_dim_;
    data_ = data_knn.data_;
    k_ = data_knn.k_;
  }

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
  CenterFinder() : DPCComputer() {}

  virtual ~CenterFinder() {}

  virtual std::set<int>
  operator()(std::vector<T> &densities, std::vector<T> &re_weighted_densities,
             std::vector<std::pair<uint32_t, double>> &dep_ptrs) = 0;
};

// Now, apply the same constructor pattern for the other classes:

template <typename T> class NoiseFinder : public DPCComputer {
public:
  NoiseFinder() : DPCComputer() {}

  virtual ~NoiseFinder() {}

  virtual std::set<int> operator()(std::vector<T> &densities,
                                   std::vector<T> &re_weighted_densities) = 0;
};

template <typename T> class ClusterAssigner : public DPCComputer {
public:
  ClusterAssigner() : DPCComputer() {}

  virtual std::vector<int>
  operator()(std::vector<T> &densities, std::vector<T> &re_weighted_densities,
             std::vector<std::pair<uint32_t, double>> &dep_ptrs,
             std::set<int> &centers) = 0;
};

template <typename T> class ClusterMerger : public DPCComputer {
public:
  ClusterMerger() : DPCComputer() {}

  virtual std::vector<int>
  operator()(std::vector<int> &current_cluster, std::vector<T> &densities,
             std::vector<T> &re_weighted_densities,
             std::vector<std::pair<uint32_t, double>> &dep_ptrs) = 0;
};

void dpc(const unsigned K, const unsigned L, const unsigned Lnn,
         RawDataset raw_data, float density_cutoff, float distance_cutoff,
         float center_density_cutoff, const std::string &output_path,
         const std::string &decision_graph_path, const unsigned Lbuild,
         const unsigned max_degree, const float alpha,
         const unsigned num_clusters, Method method, GraphType graph_type);

} // namespace DPC
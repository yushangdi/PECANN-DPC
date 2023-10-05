#pragma once

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

#include "ParlayANN/algorithms/utils/NSGDist.h"

#include "IO.h"
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
  std::optional<std::pair<int, double>>
  get_dep_ptr(int i, const std::vector<double> &densities) const {
    double d_i = densities[i];
    for (size_t j = 1; j < k_; ++j) {
      int id = knn_[i * k_ + j].first;
      double d_j = densities[id];
      if (d_j > d_i || (d_i == d_j && id > i)) {
        return knn_[i * k_ + j];
      }
    }
    return std::nullopt;
  }
};

// Base class for other DPCComputers.
class DPCComputer {
public:
  void initialize(const DatasetKnn &data_knn) {
    knn_ = data_knn.knn_;
    D_ = data_knn.D_;
    num_data_ = data_knn.num_data_;
    data_dim_ = data_knn.data_dim_;
    aligned_dim_ = data_knn.aligned_dim_;
    data_ = data_knn.data_;
    k_ = data_knn.k_;
  }

protected:
  const std::pair<int, double> *knn_;
  const Distance *D_;
  size_t num_data_;
  size_t data_dim_;
  size_t aligned_dim_;
  const float *data_;
  int k_;

  DPCComputer(const DatasetKnn &data_knn)
      : knn_(data_knn.knn_), D_(data_knn.D_), num_data_(data_knn.num_data_),
        data_dim_(data_knn.data_dim_), aligned_dim_(data_knn.aligned_dim_),
        data_(data_knn.data_), k_(data_knn.k_) {}

  DPCComputer() {}

  virtual ~DPCComputer() {} // Virtual destructor
};

class DensityComputer : public DPCComputer {
public:
  // Here we're passing the necessary arguments to the base class constructor
  DensityComputer() : DPCComputer() {}

  // Return the density.
  virtual std::vector<double> operator()() = 0;

  // Reweight the density of each point in $v$ based on knn.
  virtual std::vector<double>
  reweight_density(const std::vector<double> &densities) = 0;
};

template <typename T> class CenterFinder : public DPCComputer {
public:
  CenterFinder() : DPCComputer() {}

  ~CenterFinder() {}

  virtual std::set<int>
  operator()(const std::vector<T> &densities,
             const std::vector<T> &re_weighted_densities,
             const std::set<int> &noise_pts,
             const std::vector<std::pair<int, double>> &dep_ptrs) = 0;
};

// Now, apply the same constructor pattern for the other classes:

template <typename T> class NoiseFinder : public DPCComputer {
public:
  NoiseFinder() : DPCComputer() {}

  ~NoiseFinder() {}

  virtual std::set<int> operator()(std::vector<T> &densities,
                                   std::vector<T> &re_weighted_densities) = 0;
};

template <typename T> class ClusterAssigner : public DPCComputer {
public:
  ClusterAssigner() : DPCComputer() {}

  virtual std::vector<int>
  operator()(const std::vector<T> &densities,
             const std::vector<T> &re_weighted_densities,
             const std::set<int> &noise_pts,
             const std::vector<std::pair<int, double>> &dep_ptrs,
             const std::set<int> &centers) = 0;
};

template <typename T> class ClusterMerger : public DPCComputer {
public:
  ClusterMerger() : DPCComputer() {}

  virtual std::vector<int>
  operator()(std::vector<int> &current_cluster, std::vector<T> &densities,
             std::vector<T> &re_weighted_densities,
             std::vector<std::pair<int, double>> &dep_ptrs) = 0;
};
} // namespace DPC
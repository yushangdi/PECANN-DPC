#pragma once

#include "computers.h"
#include <set>
#include <utility>
#include <vector>

namespace DPC {

// Construct graph for `data`. Parameters:
//   raw_data: raw input data points corresponding to data.
//   D: distance computation method
//   Lbuild: the buffer size parameter
template <class T>
parlay::sequence<Tvec_point<T> *>
construct_graph(const RawDataset &raw_data, ParsedDataset &data,
                const unsigned Lbuild, const float alpha, const int max_degree,
                const int num_clusters, Distance *D,
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
            Distance *D);

// Compute the k-nearest neighbors of points in `raw_data` using bruteforce
// method. Parameters:
//   raw_data: raw input data points corresponding to data.
//   D: distance computation method
//   K: the k in knn
std::vector<std::pair<int, double>>
compute_knn_bruteforce(const RawDataset &raw_data, const unsigned K,
                       const Distance *D);

// Compute the dependet point of points in `graph` with densities above
// `density_cutoff` using graph-based method. The maximum density point has
// dep_ptr data_num. Parameters:
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
                parlay::sequence<Tvec_point<T>> &points,
                const DatasetKnn &data_knn, const RawDataset &raw_data,
                const std::vector<double> &densities,
                const std::set<int> &noise_pts, Distance *D, unsigned L,
                int round_limit = -1);

// Compute the dependet point of `raw_data` with densities above
// `density_cutoff` using bruteforce method. Parameters:
//   raw_data: input data points.
//   densities: densities of data pints.
//   density_cutoff: points with densities below this threshold are considered
//   noise points, and their dependent points are not computed.
//   D: distance computation method
std::vector<std::pair<int, double>>
compute_dep_ptr_bruteforce(const RawDataset &raw_data,
                           const DatasetKnn &data_knn,
                           const std::vector<double> &densities,
                           const std::set<int> &noise_pts, Distance *D);

template <typename T>
class KthDistanceDensityComputer : public DensityComputer<T> {
public:
  // Here we're passing the necessary arguments to the base class constructor
  KthDistanceDensityComputer() : DensityComputer<T>() {}

  // Return the density.
  std::vector<double>
  operator()(parlay::sequence<Tvec_point<T> *> &graph) override;

  // Reweight the density of each point in $v$ based on knn.
  std::vector<double>
  reweight_density(const std::vector<double> &densities) override;
};

template <typename T> class ThresholdCenterFinder : public CenterFinder<T> {
private:
  double delta_threshold_;
  double density_threshold_;

public:
  ThresholdCenterFinder(double delta_threshold, double density_threshold)
      : CenterFinder<T>(), delta_threshold_(delta_threshold),
        density_threshold_(density_threshold) {}

  ~ThresholdCenterFinder() {}

  std::set<int>
  operator()(const std::vector<T> &densities,
             const std::vector<T> &re_weighted_densities,
             const std::set<int> &noise_pts,
             const std::vector<std::pair<int, double>> &dep_ptrs) override;
};

template <typename T> class UFClusterAssigner : public ClusterAssigner<T> {

public:
  UFClusterAssigner() : ClusterAssigner<T>() {}

  ~UFClusterAssigner() {}

  std::vector<int>
  operator()(const std::vector<T> &densities,
             const std::vector<T> &re_weighted_densities,
             const std::vector<std::pair<int, double>> &dep_ptrs,
             const std::set<int> &centers) override;
};

} // namespace DPC
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

// Compute the dependet point of points in `graph` not in
// `noise_pts` using graph-based method. Points without dependent point have
// dep_ptr data_num and distance std::numeric_limits<double>::max(). Parameters:
//   graph: input data points. They should have neighbors computed already.
//   raw_data: raw input data points corresponding to data.
//   densities: densities of data pints.
//   noise_pts: noisy points
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

// Compute the dependet point of `raw_data` not in
// `noise_pts` using bruteforce method.Points without dependent point have
// dep_ptr data_num and distance std::numeric_limits<double>::max(). Parameters:
//   raw_data: input data points.
//   densities: densities of data pints.
//   noise_pts: noisy points
//   D: distance computation method
std::vector<std::pair<int, double>>
compute_dep_ptr_bruteforce(const RawDataset &raw_data,
                           const DatasetKnn &data_knn,
                           const std::vector<double> &densities,
                           const std::set<int> &noise_pts, Distance *D);

// D. Floros, T. Liu, N. Pitsianis and X. Sun, "Sparse Dual of the Density Peaks
// Algorithm for Cluster Analysis of High-dimensional Data," 2018 IEEE High
// Performance extreme Computing Conference (HPEC), 2018, pp. 1-14,
// doi: 10.1109/HPEC.2018.8547519
class KthDistanceDensityComputer : public DensityComputer {
public:
  KthDistanceDensityComputer() : DensityComputer() {}

  // Return the density. 1/ the distance to kth nearest neighbor.
  std::vector<double> operator()() override;

  // Return empty vector
  std::vector<double>
  reweight_density(const std::vector<double> &densities) override;
};

// https://ieeexplore.ieee.org/document/8765754
// Enhancing density peak clustering via density normalization
class NormalizedDensityComputer : public DensityComputer {
public:
  NormalizedDensityComputer() : DensityComputer() {}

  // Return the density. 1/ the distance to kth nearest neighbor.
  std::vector<double> operator()() override;

  // Re-weighted rho' = rho / (average rho among knn).
  std::vector<double>
  reweight_density(const std::vector<double> &densities) override;
};

// Centers are points with density >=  density_threshold_ and distance >=
// delta_threshold_ and are not noisy points.
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

// TODO (shangdi): add center finder by product of rho*delta and rho' * delta

// Non-center points merge with their dependent points using union-find.
template <typename T> class UFClusterAssigner : public ClusterAssigner<T> {

public:
  UFClusterAssigner() : ClusterAssigner<T>() {}

  ~UFClusterAssigner() {}

  std::vector<int>
  operator()(const std::vector<T> &densities,
             const std::vector<T> &re_weighted_densities,
             const std::set<int> &noise_pts,
             const std::vector<std::pair<int, double>> &dep_ptrs,
             const std::set<int> &centers) override;
};

} // namespace DPC
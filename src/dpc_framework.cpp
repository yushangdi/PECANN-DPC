#include "dpc_framework.h"
#include "dpc_components.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <string.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "IO.h"

#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

#include "ParlayANN/algorithms/utils/NSGDist.h"

#include "bruteforce.h"
#include "utils.h"

// #include <boost/json.hpp>

namespace DPC {

ClusteringResult dpc_framework(
    const unsigned K, const unsigned L, const unsigned Lnn, RawDataset raw_data,
    const std::shared_ptr<CenterFinder<double>> &center_finder,
    std::shared_ptr<DPC::DensityComputer> &density_computer,
    const std::string &output_path, const std::string &decision_graph_path,
    const unsigned Lbuild, const unsigned max_degree, const float alpha,
    const unsigned num_clusters, Method method, GraphType graph_type) {

  parlay::internal::timer t("DPC");
  std::unordered_map<std::string, double> output_metadata;

  using T = float;
  Distance *D = new Euclidian_Distance();

  // Graph Construction
  ParsedDataset parsed_data;
  parlay::sequence<Tvec_point<T> *> graph;
  if (graph_type != GraphType::BruteForce) {
    graph = construct_graph<T>(raw_data, parsed_data, Lbuild, alpha, max_degree,
                               num_clusters, D, graph_type);
  }

  output_metadata["Built index time"] = t.next_time();

  // Compute knn
  // Use K + 1 since we want to find the point itself + its k nearest neighbors
  std::vector<std::pair<int, double>> knn;
  if (graph_type == GraphType::BruteForce) {
    knn = compute_knn_bruteforce(raw_data, K + 1, D);
  } else {
    knn = compute_knn(graph, raw_data, K + 1, Lnn, D);
  }
  DatasetKnn dataset_knn(raw_data, D, K + 1, knn);

  output_metadata["Find knn time"] = t.next_time();

  // Compute density
  density_computer->initialize(dataset_knn);
  auto densities = (*density_computer)();
  auto reweighted_densities = density_computer->reweight_density(densities);
  std::set<int> noise_points;

  output_metadata["Compute density time"] = t.next_time();

  // Compute denpendent points
  std::vector<std::pair<int, double>> dep_ptrs;
  if (graph_type == GraphType::BruteForce) {
    dep_ptrs = compute_dep_ptr_bruteforce(raw_data, dataset_knn, densities,
                                          noise_points, D);
  } else {
    dep_ptrs = compute_dep_ptr(graph, parsed_data.points, dataset_knn, raw_data,
                               densities, noise_points, D, L,
                               /* round_limit = */ 4);
  }

  output_metadata["Compute dependent points time"] = t.next_time();

  // Compute centers
  center_finder->initialize(dataset_knn);
  auto centers =
      (*center_finder)(densities, reweighted_densities, noise_points, dep_ptrs);

  // Compute noises, skipping this step for now.

  // Assign clusters
  auto cluster_assigner = UFClusterAssigner<double>();
  cluster_assigner.initialize(dataset_knn);
  auto cluster = cluster_assigner(densities, reweighted_densities, noise_points,
                                  dep_ptrs, centers);

  output_metadata["Find clusters time"] = t.next_time();

  // Merge Clusters, skipping this step for now.

  // Output results
  output(densities, cluster, dep_ptrs, output_path, decision_graph_path);

  output_metadata["Output time"] = t.next_time();

  std::set<int> unique_cluster_ids(cluster.begin(), cluster.end());
  std::cout << "Num. cluster = " << unique_cluster_ids.size() << std::endl;

  output_metadata["Total time"] = t.total_time();

  return {output_metadata, cluster};
}

} // namespace DPC

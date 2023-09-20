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

bool report_stats = true;

namespace DPC {

void dpc(const unsigned K, const unsigned L, const unsigned Lnn,
         RawDataset raw_data, float density_cutoff, float distance_cutoff,
         float center_density_cutoff, const std::string &output_path,
         const std::string &decision_graph_path, const unsigned Lbuild,
         const unsigned max_degree, const float alpha,
         const unsigned num_clusters, Method method, GraphType graph_type) {
  using T = float;
  Distance *D = new Euclidian_Distance();

  // Graph Construction
  ParsedDataset parsed_data;
  parlay::sequence<Tvec_point<T> *> graph;
  if (graph_type != GraphType::BruteForce) {
    graph = construct_graph(raw_data, parsed_data, Lbuild, alpha, max_degree,
                            num_clusters, D, graph_type);
  }

  // Compute knn
  std::vector<std::pair<int, double>> knn;
  if (graph_type == GraphType::BruteForce) {
    knn = compute_knn_bruteforce(raw_data, K, D);
  } else {
    knn = compute_knn(graph, raw_data, K, Lnn, D);
  }
  DatasetKnn dataset_knn(raw_data, D, K, knn);

  // Compute density
  auto density_computer = KthDistanceDensityComputer<T>();
  density_computer.initialize(dataset_knn);
  auto densities = density_computer();
  auto reweighted_densities = density_computer.reweight_density(densities);
  std::set<int> noise_points;

  // Compute denpendent points
  std::vector<std::pair<int, double>> dep_ptrs;
  if (graph_type == GraphType::BruteForce) {
    dep_ptrs = compute_dep_ptr_bruteforce(raw_data, densities, noise_points, D);
  } else {
    dep_ptrs = compute_dep_ptr(graph, raw_data, densities, noise_points, D, L,
                              /* round_limit = */ 4);
  }

  // Compute centers
  auto center_finder =
      ThresholdCenterFinder<T>(distance_cutoff, center_density_cutoff);
  center_finder.initialize(dataset_knn);
  auto centers = center_finder(densities, reweighted_densities, dep_ptrs);

  // Compute noises, skipping this step for now.

  // Assign clusters
  auto cluster_assigner = UFClusterAssigner();
  cluster_assigner.initialize(dataset_knn);
  auto cluster =
      cluster_assigner(densities, reweighted_densities, dep_ptrs, centers);

  // Merge Clusters, skipping this step for now.

  // Output results
  output(densities, cluster, dep_ptrs, output_path, decision_graph_path);
  std::set<int> unique_cluster_ids(cluster.begin(), cluster.end());
  std::cout << "Num. cluster = " << unique_cluster_ids.size() << std::endl;
}

} // namespace DPC

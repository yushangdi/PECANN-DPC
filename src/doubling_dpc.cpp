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

#include "IO.h"
#include "bruteforce.h"
#include "union_find.h"
#include "utils.h"

namespace DPC {

// v, i, densities, data_aligned_dim, Lnn, index
template <class T>
std::pair<uint32_t, double>
compute_dep_ptr(parlay::sequence<Tvec_point<T> *> &data, std::size_t query_id,
                const std::vector<T> &densities, const size_t data_dim,
                unsigned &L, Distance *D, int round_limit = -1) {
  // if(L*4 > densities.size()) return densities.size(); // why?

  parlay::sequence<Tvec_point<T> *> start_points;
  start_points.push_back(data[query_id]);

  uint32_t dep_ptr;
  float minimum_dist;

  if (round_limit == -1) {
    round_limit = densities.size(); // effectively no limit on round.
  }

  for (int round = 0; round < round_limit; ++round) {
    auto [pairElts, dist_cmps] =
        beam_search<T>(data[query_id], data, start_points, L, data_dim, D);
    auto [beamElts, visitedElts] = pairElts;

    double query_density = densities[query_id];
    T *query_ptr = data[query_id]->coordinates.begin();
    minimum_dist = std::numeric_limits<float>::max();
    dep_ptr = densities.size();
    for (unsigned i = 0; i < beamElts.size(); i++) {
      const auto [id, dist] = beamElts[i];
      if (id == query_id)
        continue;
      // if(id == densities.size()) break;
      if (densities[id] > query_density ||
          (densities[id] == query_density && id > query_id)) {
        if (dist < minimum_dist) {
          minimum_dist = dist;
          dep_ptr = id;
        }
      }
    }
    if (dep_ptr != densities.size()) {
      break;
    }
    L *= 2;
  }

  // if(dep_ptr == densities.size()){
  // 	L *= 2;
  // 	return compute_dep_ptr(data, query_id, densities, data_aligned_dim, L,
  // D);
  // }
  return {dep_ptr, sqrt(minimum_dist)};
}

// v, i, densities, data_aligned_dim, Lnn, index
// template<class T>
// std::pair<uint32_t, double>
// compute_dep_ptr_blind_probe(parlay::sequence<Tvec_point<T>*> data,
// std::size_t query_id, const std::vector<T>& densities,
// const size_t data_aligned_dim, unsigned& L, Distance* D){
// 	// if(L*4 > densities.size()) return densities.size(); // why?

// 	parlay::sequence<Tvec_point<T>*> start_points;
// 	start_points.push_back(data[query_id]);
// 	auto [pairElts, dist_cmps] = beam_search_blind_probe<T,
// T>(data[query_id], data, densities,
// start_points, L, data_aligned_dim, D); 	auto [beamElts, visitedElts] =
// pairElts;

// 	double query_density = densities[query_id];
// 	T* query_ptr = data[query_id]->coordinates.begin();
// 	float minimum_dist = std::numeric_limits<float>::max();
// 	uint32_t dep_ptr = densities.size();
// 	for(unsigned i=0; i<beamElts.size(); i++){
// 		const auto [id, dist] = beamElts[i];
// 		if (id == query_id) continue;
// 		// if(id == densities.size()) break;
// 		if(densities[id] > query_density || (densities[id] ==
// query_density
// && id > query_id)){ 			if(dist < minimum_dist){
// minimum_dist = dist; 				dep_ptr = id;
// 			}
// 		} else {
// 			std::cout << "Internal error: blind probe retuned
// invalid points \n.";
// 		}
// 	}
// 	if(dep_ptr == densities.size()){
// 		L *= 2;
// 		return compute_dep_ptr_blind_probe(data, query_id, densities,
// data_aligned_dim, L, D);
// 	}
// 	return {dep_ptr, minimum_dist};
// }

template <class T>
void compute_densities(parlay::sequence<Tvec_point<T> *> &v,
                       std::vector<T> &densities, const unsigned L,
                       const unsigned K, const size_t data_num,
                       const size_t data_dim, Distance *D,
                       const GraphType graph_type) {
  auto beamSizeQ = L;
  std::atomic<int> num_bruteforce = 0;
  // potential TODO: add a random option. The graph search order is random if
  // random flag is on. Should be on for pyDNN should this random just be on for
  // all searches? size_t n = v.size();
  // parlay::random_generator gen;
  // std::uniform_int_distribution<long> dis(0, n-1);
  // auto indices = parlay::tabulate(q.size(), [&](size_t i) {
  //   auto r = gen[i];
  //   return dis(r);
  // });

  parlay::parallel_for(0, data_num, [&](size_t i) {
    // parlay::sequence<int> neighbors = parlay::sequence<int>(k);
    // what is cut and limit?
    parlay::sequence<Tvec_point<T> *> start_points;
    start_points.push_back(v[i]);
    auto [pairElts, dist_cmps] =
        beam_search(v[i], v, start_points, beamSizeQ, data_dim, D, K + 1);
    auto [beamElts, visitedElts] = pairElts;
    T distance;
    if (beamElts.size() <
        K + 1) { // found less than K + 1 neighbors during the search
      std::vector<float> dists(data_num);
      parlay::parallel_for(0, data_num, [&](size_t j) {
        dists[j] = D->distance(v[i]->coordinates.begin(),
                               v[j]->coordinates.begin(), data_dim);
      });
      std::nth_element(dists.begin(), dists.begin() + K, dists.end());
      distance = dists[K];
      std::atomic_fetch_add(&num_bruteforce, 1);
    } else {
      auto less = [&](id_dist a, id_dist b) {
        return a.second < b.second ||
               (a.second == b.second && a.first < b.first);
      };
      auto sorted_nn = parlay::sort(beamElts, less);
      distance = sorted_nn[K].second;
    }
    if (distance <= 0) {
      densities[i] = std::numeric_limits<T>::max();
    } else {
      densities[i] = 1.0 / sqrt(distance);
    }
  });
  std::cout << "num bruteforce " << num_bruteforce.load() << std::endl;
}

void approx_dpc(const unsigned K, const unsigned L, const unsigned Lnn,
                ParsedDataset data, float density_cutoff, float distance_cutoff,
                float center_density_cutoff, const std::string &output_path,
                const std::string &decision_graph_path, const unsigned Lbuild,
                const unsigned max_degree, const float alpha,
                const unsigned num_clusters, Method method,
                GraphType graph_type) {
  // using std::chrono::high_resolution_clock;
  // using std::chrono::duration_cast;
  // using std::chrono::duration;
  // using std::chrono::microseconds;
  using T = float;
  parlay::internal::timer t("DPC");

  Distance *D = new Euclidian_Distance();

  add_null_graph(data.points, max_degree);
  auto v = parlay::tabulate(
      data.size, [&](size_t i) -> Tvec_point<T> * { return &data.points[i]; });
  t.next("Load data.");

  if (graph_type == GraphType::Vamana) {
    using findex = knn_index<T>;
    findex I(max_degree, Lbuild, alpha, data.data_dim, D);
    parlay::sequence<int> inserts = parlay::tabulate(
        v.size(), [&](size_t i) { return static_cast<int>(i); });
    I.build_index_multiple_starts(
        v, inserts, Lbuild - 1); // threshold cannot exceed Lbuild-1
  } else if (graph_type == GraphType::pyNNDescent) {
    using findex = pyNN_index<T>;
    // .05: terminate graph construction if < .05 neighbor change is happening.
    findex I(max_degree, data.data_dim, .05, D);
    auto cluster_size = Lbuild;
    I.build_index(v, cluster_size, num_clusters, alpha);
  } else if (graph_type == GraphType::HCNNG) {
    using findex = hcnng_index<T>;
    findex I(max_degree, data.data_dim, D);
    auto cluster_size = Lbuild;
    I.build_index(v, num_clusters, cluster_size);
  } else {
    std::cout << "Error: method not implemented " << std::endl;
    exit(1);
  }

  double build_time = t.next_time();
  report(build_time, "Built index");

  if (report_stats) {
    auto [avg_deg, max_deg] = graph_stats(v);
    std::cout << "Index built with average degree " << avg_deg
              << " and max degree " << max_deg << std::endl;
    t.next("stats");
  }

  std::vector<T> densities(data.size);
  compute_densities(v, densities, L, K, data.size, data.data_dim, D,
                    graph_type);
  double density_time = t.next_time();
  report(density_time, "Compute density");

  // sort in desending order
  // auto sorted_points= parlay::sequence<unsigned>::from_function(data_num,
  // [](unsigned i){return i;}); parlay::sort_inplace(sorted_points,
  // [&densities](unsigned i, unsigned j){ 	return densities[i] >
  // densities[j]
  // || (densities[i] == densities[j] && i > j);
  // });
  // auto max_point_id = sorted_points[0];
  // unsigned threshold = log(data_num);

  Tvec_point<T> **max_density_point =
      parlay::max_element(v, [&densities](Tvec_point<T> *a, Tvec_point<T> *b) {
        if (densities[a->id] == densities[b->id]) {
          return a->id < b->id;
        }
        return densities[a->id] < densities[b->id];
      });
  auto max_point_id = max_density_point[0]->id;
  unsigned threshold = 0;

  std::vector<std::pair<uint32_t, double>> dep_ptrs(data.size);
  // dep_ptrs[max_point_id] = {data_num, -1};
  parlay::parallel_for(0, data.size, [&](size_t i) {
    dep_ptrs[i] = {data.size, -1};
  });
  auto unfinished_points = parlay::sequence<unsigned>::from_function(
      data.size, [](unsigned i) { return i; });
  unfinished_points = parlay::filter(unfinished_points, [&](size_t i) {
    return i != max_point_id &&
           densities[i] > density_cutoff; // skip noise points
  });
  // compute the top log n density points using bruteforce
  //  std::cout << "threshold: " << threshold << std::endl;
  //  bruteforce_dependent_point(0, data_num, sorted_points, points, densities,
  //  dep_ptrs, density_cutoff, D, data_dim);
  //  bruteforce_dependent_point(threshold, data_num, sorted_points, points,
  //  densities, dep_ptrs, density_cutoff, D, data_dim);

  std::vector<unsigned> num_rounds(
      data.size, Lnn); // the L used when dependent point is found.
  if (method == Method::Doubling) {
    int round_limit = 4;
    int prev_number = std::numeric_limits<int>::max();
    while (
        unfinished_points.size() > 300 &&
        prev_number >
            unfinished_points
                .size()) { // stop if unfinished_points number does not decrease
      prev_number = unfinished_points.size();
      parlay::parallel_for(0, unfinished_points.size(), [&](size_t j) {
        // auto i = sorted_points[j];
        auto i = unfinished_points[j];
        dep_ptrs[i] = compute_dep_ptr(v, i, densities, data.data_dim,
                                      num_rounds[i], D, round_limit);
        // }
      });
      unfinished_points =
          parlay::filter(unfinished_points, [&dep_ptrs, &data](size_t i) {
            return dep_ptrs[i].first == data.size;
          });
      std::cout << "number: " << unfinished_points.size() << std::endl;
    }
    std::cout << "bruteforce number: " << unfinished_points.size() << std::endl;
    bruteforce_dependent_point_all(data.size, unfinished_points, data.points,
                                   densities, dep_ptrs, D, data.data_dim);
    // } else if (method == Method::BlindProbe){
    // 	parlay::parallel_for(threshold, data_num, [&](size_t j) {
    // 		// auto i = sorted_points[j];
    // 		auto i = unfinished_points[j];
    // 		if (i != max_point_id && densities[i] > density_cutoff){ // skip
    // noise
    // points 			unsigned Li = Lnn; dep_ptrs[i] =
    // compute_dep_ptr_blind_probe(v, i, densities, data_aligned_dim, Li, D);
    // num_rounds[i] = Li;
    // 		}
    // 	});
  } else {
    std::cout << "Error: method not implemented " << std::endl;
    exit(1);
  }
  double dependent_time = t.next_time();
  report(dependent_time, "Compute dependent points");

  auto cluster = cluster_points(densities, dep_ptrs, density_cutoff,
                                distance_cutoff, center_density_cutoff);
  double cluster_time = t.next_time();
  report(cluster_time, "Find clusters");
  report(build_time + density_time + dependent_time + cluster_time, "Total");

  output(densities, cluster, dep_ptrs, output_path, decision_graph_path);
  std::set<int> unique_cluster_ids(cluster.begin(), cluster.end());
  std::cout << "Num. cluster = " << unique_cluster_ids.size() << std::endl;
  writeVectorToFile(num_rounds, "results/num_rounds.txt");
}

std::unordered_map<std::string, double>
dpc(const unsigned K, const unsigned L, const unsigned Lnn, RawDataset raw_data,
    float density_cutoff, float distance_cutoff, float center_density_cutoff,
    const std::string &output_path, const std::string &decision_graph_path,
    const unsigned Lbuild, const unsigned max_degree, const float alpha,
    const unsigned num_clusters, Method method, GraphType graph_type) {

  time_reports.clear();

  ParsedDataset parsed_data(raw_data);

  std::cout << "output_file=" << output_path << "\n";
  std::cout << "decision_graph_path=" << decision_graph_path << "\n";
  std::cout << "density_cutoff=" << density_cutoff << "\n";
  std::cout << "center_density_cutoff=" << center_density_cutoff << "\n";
  std::cout << "dist_cutoff=" << distance_cutoff << "\n";
  std::cout << "num_thread: " << parlay::num_workers() << std::endl;

  if (graph_type == GraphType::BruteForce) {
    std::cout << "method= brute force\n";
    DPC::dpc_bruteforce(K, parsed_data, density_cutoff, distance_cutoff,
                        center_density_cutoff, output_path,
                        decision_graph_path);
    return time_reports;
  }

  std::cout << "graph_type=" << graph_type << std::endl;
  std::cout << "method=" << method << std::endl;
  std::cout << "K=" << K << "\n";
  std::cout << "L=" << L << "\n";
  std::cout << "Lnn=" << Lnn << "\n";
  std::cout << "Lbuild=" << Lbuild << "\n";
  std::cout << "max_degree=" << max_degree << "\n";
  if (graph_type != GraphType::HCNNG) {
    std::cout << "alpha=" << alpha << "\n";
  }
  if (graph_type == GraphType::pyNNDescent || graph_type == GraphType::HCNNG) {
    std::cout << "num_clusters=" << num_clusters << "\n";
    if (Lbuild < 8) {
      std::cerr << "Please use Lbuild >= 8 for pyNNDescent and HCNNG\n";
      std::cerr << "Lbuild = " << Lbuild << std::endl;
      exit(1);
    }
  }

  approx_dpc(K, L, Lnn, parsed_data, density_cutoff, distance_cutoff,
             center_density_cutoff, output_path, decision_graph_path, Lbuild,
             max_degree, alpha, num_clusters, method, graph_type);

  return time_reports;
}

} // namespace DPC
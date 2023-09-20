#include "dpc_components.h"
#include "bruteforce.h"

#include <set>
#include <utility>
#include <vector>

namespace DPC {

template <class T>
parlay::sequence<Tvec_point<T> *>
construct_graph(const RawDataset &raw_data, ParsedDataset &data,
                const unsigned Lbuild, const float alpha, const int max_degree,
                const int num_clusters, const Distance *D,
                const GraphType graph_type) {
  parlay::internal::timer t("DPC");
  data = ParsedDataset(raw_data);
  add_null_graph(data.points, max_degree);
  auto v = parlay::tabulate(
      data_num, [&](size_t i) -> Tvec_point<T> * { return &data.points[i]; });
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
  return v;
}

template <class T>
std::vector<std::pair<int, double>>
compute_knn(parlay::sequence<Tvec_point<T> *> &graph,
            const RawDataset &raw_data, const unsigned K, const unsigned L,
            const Distance *D) {
  auto beamSizeQ = L;
  std::atomic<int> num_bruteforce = 0;
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
  std::vector<std::pair<int, double>> knn(K * data_num);
  parlay::parallel_for(0, data_num, [&](size_t i) {
    parlay::sequence<Tvec_point<T> *> start_points;
    start_points.push_back(v[i]);
    auto [pairElts, dist_cmps] =
        beam_search(v[i], v, start_points, beamSizeQ, data_dim, D, K);
    auto [beamElts, visitedElts] = pairElts;
    auto less = [&](id_dist a, id_dist b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    if (beamElts.size() <= K) { // found less than K neighbors during the search
      std::vector<std::pair<int, double>> dists(data_num);
      parlay::parallel_for(0, data_num, [&](size_t j) {
        dists[j] =
            std::make_pair(j, D->distance(v[i]->coordinates.begin(),
                                          v[j]->coordinates.begin(), data_dim));
      });
      std::nth_element(dists.begin(), dists.begin() + K, dists.end(), less);
      std::atomic_fetch_add(&num_bruteforce, 1);
      std::sort(dists.begin(), dists.begin() + K, less);
      for (int kth = 0; kth < K; ++kth) {
        knn[i * K + kth] = dists[kth];
      }
    } else {
      auto sorted_nn = parlay::sort(beamElts, less);
      for (int kth = 0; kth < K; ++kth) {
        knn[i * K + kth] = sorted_nn[kth];
      }
    }
  });
  std::cout << "num bruteforce " << num_bruteforce.load() << std::endl;
  return knn;
}

template <class T>
std::vector<std::pair<int, double>>
compute_knn_bruteforce(const RawDataset &raw_data, const unsigned K,
                       const Distance *D) {
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
  int aligned_dim = raw_data.aligned_dim;
  float *data = raw_data.data;
  auto less = [&](id_dist a, id_dist b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };
  std::vector<std::pair<int, double>> knn(K * data_num);
  parlay::parallel_for(0, data_num, [&](size_t i) {
    std::vector<std::pair<int, double>> dists(data_num);
    parlay::parallel_for(0, data_num, [&](size_t j) {
      dists[j] = std::make_pair(j, D->distance(data[i], data[j], data_dim));
    });
    std::nth_element(dists.begin(), dists.begin() + K, dists.end(), less);
    std::sort(dists.begin(), dists.begin() + K, less);
    for (int kth = 0; kth < K; ++kth) {
      knn[i * K + kth] = dists[kth];
    }
  });
  return knn;
}

template <class T>
std::pair<int, double>
compute_dep_ptr(parlay::sequence<Tvec_point<T> *> &data, std::size_t query_id,
                const std::vector<T> &densities, const size_t data_dim,
                unsigned &L, Distance *D, int round_limit = -1) {
  parlay::sequence<Tvec_point<T> *> start_points;
  start_points.push_back(data[query_id]);

  int dep_ptr;
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

  return {dep_ptr, sqrt(minimum_dist)};
}

template <class T>
std::vector<std::pair<int, double>>
compute_dep_ptr(parlay::sequence<Tvec_point<T> *> &graph,
                const DatasetKnn &data_knn, const RawDataset &raw_data,
                const std::vector<T> &densities, const std::set<int> &noise_pts,
                const Distance *D, unsigned L, int round_limit) {
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
  int aligned_dim = raw_data.aligned_dim;
  std::vector<bool> finished(data_num, false);
  Tvec_point<T> **max_density_point =
      parlay::max_element(v, [&densities](Tvec_point<T> *a, Tvec_point<T> *b) {
        if (densities[a->id] == densities[b->id]) {
          return a->id < b->id;
        }
        return densities[a->id] < densities[b->id];
      });
  auto max_point_id = max_density_point[0]->id;
  finished[max_point_id] = true;
  unsigned threshold = 0;

  std::vector<std::pair<int, double>> dep_ptrs(data_num);
  parlay::parallel_for(0, data_num, [&](size_t i) {
    float m_dist = std::numeric_limits<float>::max();
    size_t id = data_num;
    if (noise_pts.contains(i)) { // skip noise points
      // search within knn
      auto dep_pt = data_knn.get_dep_ptr(i, densities);
      if (dep_pt.has_value()) {
        finished[i] = true;
        dep_ptrs =
            std::make_pair(dep_pt.value().first, sqrt(dep_pt.value().second));
      }
    } else {
      dep_ptrs[i] = {data_num, -1};
    }
  });
  auto unfinished_points = parlay::sequence<unsigned>::from_function(
      data_num, [](unsigned i) { return i; });
  unfinished_points = parlay::filter(unfinished_points, [&](size_t i) {
    return i != max_point_id && (!noise_pts.contains(i)) &&
           (!finished[i]); // skip noise points and finished points
  });

  std::vector<unsigned> num_rounds(
      data_num, Lnn); // the L used when dependent point is found.
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
        auto i = unfinished_points[j];
        dep_ptrs[i] = compute_dep_ptr(graph, i, densities, data_dim,
                                      num_rounds[i], D, round_limit);
        // }
      });
      unfinished_points =
          parlay::filter(unfinished_points, [&dep_ptrs, &data](size_t i) {
            return dep_ptrs[i].first == data_num;
          });
      std::cout << "number: " << unfinished_points.size() << std::endl;
    }
    std::cout << "bruteforce number: " << unfinished_points.size() << std::endl;
    bruteforce_dependent_point_all(data_num, unfinished_points, graph.points,
                                   densities, dep_ptrs, D, data_dim);
  } else {
    std::cout << "Error: method not implemented " << std::endl;
    exit(1);
  }
  return dep_ptrs;
}

template <class T>
std::vector<std::pair<int, double>>
compute_dep_ptr_bruteforce(const RawDataset &raw_data,
                           const DatasetKnn &data_knn,
                           const std::vector<T> &densities,
                           const std::set<int> &noise_pts, Distance *D) {
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
  int aligned_dim = raw_data.aligned_dim;
  float *data = raw_data.data;
  std::vector<std::pair<int, double>> dep_ptrs(data_num);
  parlay::parallel_for(0, data_num, [&](size_t i) {
    float m_dist = std::numeric_limits<float>::max();
    size_t id = data_num;
    if (noise_pts.contains(i)) { // skip noise points
      // search within knn
      auto dep_pt = data_knn.get_dep_ptr(i, densities);
      if (dep_pt.has_value()) {
        dep_ptrs =
            std::make_pair(dep_pt.value().first, sqrt(dep_pt.value().second));
      } else {
        // bruteforce
        for (size_t j = 0; j < data_num; j++) {
          if (densities[j] > densities[i] ||
              (densities[j] == densities[i] && j > i)) {
            auto dist = D->distance(data[i], data[j], data_dim);
            if (dist <= m_dist) {
              m_dist = dist;
              id = j;
            }
          }
        }
        dep_ptrs[i] = {id, sqrt(m_dist)};
      } // end else
    }
  });
  return dep_ptrs;
}

template <typename T>
std::vector<double> KthDistanceDensityComputer<T>::operator()(
    parlay::sequence<Tvec_point<T> *> &graph) {
  int data_num = this->data_num_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    densities[i] = 1.0 / sqrt(this->knn_[(i + 1) * k - 1].second);
  });
  return densities;
}

template <typename T>
std::vector<double> KthDistanceDensityComputer<T>::reweight_density(
    const std::vector<double> &densities) {
  return {};
}

template <typename T>
std::set<int> ThresholdCenterFinder<T>::operator()(
    const std::vector<T> &densities,
    const std::vector<T> &re_weighted_densities, const std::set<int> &noise_pts,
    const std::vector<std::pair<int, double>> &dep_ptrs) {
  auto data_num = densities.size();
  auto ids = parlay::delayed_seq<int>(data_num, [](size_t i) { return i; });
  auto centers_seq = parlay::filter(ids, [&](size_t i) {
    if (dep_ptrs[i].first != data_num) { // the max density point
      return false;
    }
    if (noise_pts.contains(i))
      return false;
    if (dep_ptrs[i].second >= delta_threshold_ &&
        densities[i] >= density_threshold_) {
      return true;
    }
    return false;
  });
  std::set<int> centers(centers_seq.begin(), centers_seq.end());
  return centers;
}

template <typename T>
std::vector<int> UFClusterAssigner<T>::operator()(
    const std::vector<T> &densities,
    const std::vector<T> &re_weighted_densities,
    const std::vector<std::pair<int, double>> &dep_ptrs,
    const std::set<int> &centers) {
  ParUF<int> UF(densities.size());
  parlay::parallel_for(0, densities.size(), [&](int i) {
    if (dep_ptrs[i].first != densities.size()) { // the max density point
      if (!centers.contains(i)) {
        UF.link(i, dep_ptrs[i].first);
      }
    }
  });
  std::vector<int> cluster(densities.size());
  parlay::parallel_for(0, densities.size(),
                       [&](int i) { cluster[i] = UF.find(i); });
  return cluster;
}

} // namespace DPC
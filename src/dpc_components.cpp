#include "dpc_components.h"

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

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "IO.h"
#include "bruteforce.h"
#include "union_find.h"
#include "utils.h"

namespace DPC {

template <class T>
parlay::sequence<Tvec_point<T> *>
construct_graph(const RawDataset &raw_data, ParsedDataset &data,
                const unsigned Lbuild, const float alpha, const int max_degree,
                const int num_clusters, Distance *D,
                const GraphType graph_type) {
  parlay::internal::timer t("DPC");
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
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
            Distance *D) {
  auto beamSizeQ = L;
  std::atomic<int> num_bruteforce = 0;
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
  std::vector<std::pair<int, double>> knn(K * data_num);
  parlay::parallel_for(0, data_num, [&](size_t i) {
    parlay::sequence<Tvec_point<T> *> start_points;
    start_points.push_back(graph[i]);
    auto [pairElts, dist_cmps] =
        beam_search(graph[i], graph, start_points, beamSizeQ, data_dim, D, K);
    auto [beamElts, visitedElts] = pairElts;
    auto less = [&](id_dist a, id_dist b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    if (beamElts.size() < K) { // found less than K neighbors during the search
      std::vector<std::pair<int, double>> dists(data_num);
      parlay::parallel_for(0, data_num, [&](size_t j) {
        dists[j] = std::make_pair(j, D->distance(graph[i]->coordinates.begin(),
                                                 graph[j]->coordinates.begin(),
                                                 data_dim));
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

std::vector<std::pair<int, double>>
compute_knn_bruteforce(const RawDataset &raw_data, const unsigned K,
                       const Distance *D) {
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
  int aligned_dim = raw_data.aligned_dim;
  auto less = [&](id_dist a, id_dist b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };
  std::vector<std::pair<int, double>> knn(K * data_num);
  parlay::parallel_for(0, data_num, [&](size_t i) {
    std::vector<std::pair<int, double>> dists(data_num);
    parlay::parallel_for(0, data_num, [&](size_t j) {
      dists[j] =
          std::make_pair(j, D->distance(raw_data[i], raw_data[j], data_dim));
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
                const std::vector<double> &densities, const size_t data_dim,
                unsigned &L, Distance *D, int round_limit = -1) {
  parlay::sequence<Tvec_point<T> *> start_points;
  start_points.push_back(data[query_id]);

  int dep_ptr;
  double minimum_dist;

  if (round_limit == -1) {
    round_limit = densities.size(); // effectively no limit on round.
  }

  for (int round = 0; round < round_limit; ++round) {
    auto [pairElts, dist_cmps] =
        beam_search<T>(data[query_id], data, start_points, L, data_dim, D);
    auto [beamElts, visitedElts] = pairElts;

    double query_density = densities[query_id];
    T *query_ptr = data[query_id]->coordinates.begin();
    minimum_dist = std::numeric_limits<double>::max();
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
std::vector<std::pair<int, double>> compute_dep_ptr(
    parlay::sequence<Tvec_point<T> *> &graph,
    parlay::sequence<Tvec_point<T>> &points, const DatasetKnn &data_knn,
    const RawDataset &raw_data, const std::vector<double> &densities,
    const std::set<int> &noise_pts, Distance *D, unsigned L, int round_limit) {
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
  int aligned_dim = raw_data.aligned_dim;
  std::vector<bool> finished(data_num, false);
  Tvec_point<T> **max_density_point = parlay::max_element(
      graph, [&densities](Tvec_point<T> *a, Tvec_point<T> *b) {
        if (densities[a->id] == densities[b->id]) {
          return a->id < b->id;
        }
        return densities[a->id] < densities[b->id];
      });
  auto max_point_id = max_density_point[0]->id;
  finished[max_point_id] = true;
  unsigned threshold = 0;

  std::vector<std::pair<int, double>> dep_ptrs(data_num);
  dep_ptrs[max_point_id] = {data_num, sqrt(std::numeric_limits<double>::max())};
  parlay::parallel_for(0, data_num, [&](size_t i) {
    double m_dist = std::numeric_limits<double>::max();
    size_t id = data_num;
    if (noise_pts.find(i) == noise_pts.end()) {
      // search within knn
      auto dep_pt = data_knn.get_dep_ptr(i, densities);
      if (dep_pt.has_value()) {
        finished[i] = true;
        dep_ptrs[i] =
            std::make_pair(dep_pt.value().first, sqrt(dep_pt.value().second));
      }
    } else { // skip noise points
      dep_ptrs[i] = {data_num, sqrt(std::numeric_limits<double>::max())};
    }
  });
  auto unfinished_points = parlay::sequence<unsigned>::from_function(
      data_num, [](unsigned i) { return i; });
  unfinished_points = parlay::filter(unfinished_points, [&](size_t i) {
    return i != max_point_id && (noise_pts.find(i) == noise_pts.end()) &&
           (!finished[i]); // skip noise points and finished points
  });

  std::vector<unsigned> num_rounds(
      data_num, L); // the L used when dependent point is found.
  int prev_number = std::numeric_limits<int>::max();
  while (unfinished_points.size() > 300 &&
         prev_number > unfinished_points.size()) { // stop if unfinished_points
                                                   // number does not decrease
    prev_number = unfinished_points.size();
    parlay::parallel_for(0, unfinished_points.size(), [&](size_t j) {
      auto i = unfinished_points[j];
      dep_ptrs[i] = compute_dep_ptr(graph, i, densities, data_dim,
                                    num_rounds[i], D, round_limit);
      // }
    });
    unfinished_points =
        parlay::filter(unfinished_points, [&dep_ptrs, &data_num](size_t i) {
          return dep_ptrs[i].first == data_num;
        });
    std::cout << "number: " << unfinished_points.size() << std::endl;
  }
  std::cout << "bruteforce number: " << unfinished_points.size() << std::endl;
  bruteforce_dependent_point_all(data_num, unfinished_points, points, densities,
                                 dep_ptrs, D, data_dim);
  return dep_ptrs;
}

std::vector<std::pair<int, double>>
compute_dep_ptr_bruteforce(const RawDataset &raw_data,
                           const DatasetKnn &data_knn,
                           const std::vector<double> &densities,
                           const std::set<int> &noise_pts, Distance *D) {
  int data_num = raw_data.num_data;
  int data_dim = raw_data.data_dim;
  int aligned_dim = raw_data.aligned_dim;
  std::vector<std::pair<int, double>> dep_ptrs(data_num);
  parlay::parallel_for(0, data_num, [&](size_t i) {
    double m_dist = std::numeric_limits<double>::max(); // squared distance
    size_t id = data_num;
    if (noise_pts.find(i) == noise_pts.end()) { // skip noise points
      // search within knn
      auto dep_pt = data_knn.get_dep_ptr(i, densities);
      if (dep_pt.has_value()) {
        id = dep_pt.value().first;
        m_dist = dep_pt.value().second;
      } else {
        // bruteforce
        for (size_t j = 0; j < data_num; j++) {
          if (densities[j] > densities[i] ||
              (densities[j] == densities[i] && j > i)) {
            auto dist = D->distance(raw_data[i], raw_data[j], data_dim);
            if (dist <= m_dist) {
              m_dist = dist;
              id = j;
            }
          }
        }
      } // end else
    }
    dep_ptrs[i] = {id, sqrt(m_dist)};
  });
  return dep_ptrs;
}

std::vector<double> KthDistanceDensityComputer::operator()() {
  int data_num = this->num_data_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    densities[i] = 1.0 / sqrt(this->knn_[(i + 1) * k - 1].second);
  });
  return densities;
}

std::vector<double> KthDistanceDensityComputer::reweight_density(
    const std::vector<double> &densities) {
  return {};
}

std::vector<double> NormalizedDensityComputer::operator()() {
  int data_num = this->num_data_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    densities[i] = 1.0 / sqrt(this->knn_[(i + 1) * k - 1].second);
  });
  return densities;
}

std::vector<double> NormalizedDensityComputer::reweight_density(
    const std::vector<double> &densities) {
  int data_num = this->num_data_;
  int k = this->k_;
  std::vector<double> new_densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    auto knn_densities = parlay::delayed_seq<double>(k, [&](size_t j) {
      auto neigh = this->knn_[i * k + j].first;
      return densities[neigh];
    });
    double avg_density = parlay::reduce(knn_densities) / k;
    new_densities[i] = densities[i] / avg_density;
  });
  return new_densities;
}

std::vector<double> ExpSquaredDensityComputer::operator()() {
  int data_num = this->num_data_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    auto knn_densities = parlay::delayed_seq<double>(k, [&](size_t j) {
      const double dist = this->knn_[i * k + j].second;
      return dist * dist;
    });
    double density = exp(-1.0 * parlay::reduce(knn_densities) / k);
    densities[i] = density;
  });
  return densities;
}

std::vector<double> ExpSquaredDensityComputer::reweight_density(
    const std::vector<double> &densities) {
  return std::vector<double>();
}

std::vector<double> MutualKNNDensityComputer::operator()() {
  int data_num = this->num_data_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  // knn_mapping[i] is a map from j to j's rank in i's nn.
  std::vector<std::map<int, int>> knn_mapping(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    for (int kth = 0; kth < k; ++kth) {
      // kth nearest neighbor of j is i.
      const int neigh = this->knn_[i * k + kth].first;
      knn_mapping[i][neigh] = kth;
    }
  });

  parlay::parallel_for(0, data_num, [&](int i) {
    double mnd = 0;      // mutual neighbor distance
    int ng_size = 0;     // number of mutual neighbors;
    double dist_sum = 0; // sum of euclidean distance to mutual neighbors.
    for (int kth = 0; kth < k; ++kth) {
      const auto [neigh, dist] = this->knn_[i * k + kth];
      const auto itr = knn_mapping[neigh].find(i);
      if (itr != knn_mapping[neigh].end()) {
        const int eps_pq = itr->second;
        ng_size++;
        dist_sum += sqrt(dist);
        //  add 2 because c++ start with 0 indexing
        mnd += 1.0 / (kth + eps_pq + 2);
      }
    }
    double density = std::numeric_limits<double>::infinity();
    if (dist_sum != 0) {
      density = mnd * ng_size / dist_sum;
    }
    densities[i] = density;
  });
  return densities;
}

std::vector<double> MutualKNNDensityComputer::reweight_density(
    const std::vector<double> &densities) {
  return std::vector<double>();
}

std::vector<double> RaceDensityComputer::operator()() {
  int data_num = this->num_data_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    race_sketch_.add(&this->data_[this->aligned_dim_ * i]);
  });
  parlay::parallel_for(0, data_num, [&](int i) {
    densities[i] = race_sketch_.query(&this->data_[this->aligned_dim_ * i]);
  });
  return densities;
}

std::vector<double>
RaceDensityComputer::reweight_density(const std::vector<double> &densities) {
  return {};
}


std::vector<double> SumExpDensityComputer::operator()() {
  int data_num = this->num_data_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    auto knn_densities = parlay::delayed_seq<double>(k, [&](size_t j) {
      const double dist = std::exp(-this->knn_[i * k + j].second);
      return dist * dist;
    });
    double density = parlay::reduce(knn_densities) / k;
    densities[i] = density;
  });
  return densities;
}

std::vector<double> SumExpDensityComputer::reweight_density(
    const std::vector<double> &densities) {
  return std::vector<double>();
}


std::vector<double> TopKSumDensityComputer::operator()() {
  int data_num = this->num_data_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    auto knn_densities = parlay::delayed_seq<double>(k, [&](size_t j) {
      const double dist = this->knn_[i * k + j].second;
      return dist;
    });
    double density = -parlay::reduce(knn_densities);
    densities[i] = density;
  });
  return densities;
}

std::vector<double> TopKSumDensityComputer::reweight_density(
    const std::vector<double> &densities) {
  return std::vector<double>();
}


template <typename T>
std::set<int> ThresholdCenterFinder<T>::operator()(
    const std::vector<T> &densities,
    const std::vector<T> &re_weighted_densities, const std::set<int> &noise_pts,
    const std::vector<std::pair<int, double>> &dep_ptrs) {
  auto data_num = densities.size();
  auto ids = parlay::delayed_seq<int>(data_num, [](size_t i) { return i; });
  auto centers_seq = parlay::filter(ids, [&](size_t i) {
    if (dep_ptrs[i].first == data_num) { // the max density point
      return true;
    }
    if (noise_pts.find(i) != noise_pts.end())
      return false;
    if (dep_ptrs[i].second >= dependant_dist_threshold &&
        densities[i] >= density_threshold_) {
      return true;
    }
    return false;
  });
  std::set<int> centers(centers_seq.begin(), centers_seq.end());
  return centers;
}

template <typename T>
std::set<int> ProductCenterFinder<T>::operator()(
    const std::vector<T> &densities,
    const std::vector<T> &re_weighted_densities, const std::set<int> &noise_pts,
    const std::vector<std::pair<int, double>> &dep_ptrs) {
  auto data_num = densities.size();
  assert(dep_ptrs.size() >= data_num);
  if (use_reweighted_density_) {
    assert(re_weighted_densities.size() >= data_num);
  }
  parlay::sequence<double> negative_products(data_num);
  parlay::parallel_for(0, negative_products.size(), [&](int i) {
    if (use_reweighted_density_) {
      negative_products[i] = -re_weighted_densities[i] * dep_ptrs[i].second;
    } else {
      negative_products[i] = -densities[i] * dep_ptrs[i].second;
    }
    if (dep_ptrs[i].first == densities.size()) {
      negative_products[i] = -std::numeric_limits<double>::infinity();
    }
  });
  double centroid_threshold =
      -1 * parlay::kth_smallest_copy(negative_products, num_clusters_);

  auto ids = parlay::delayed_seq<int>(data_num, [](size_t i) { return i; });
  auto centers_seq = parlay::filter(ids, [&](size_t i) {
    return -negative_products[i] > centroid_threshold;
  });
  std::set<int> centers(centers_seq.begin(), centers_seq.end());
  return centers;
}

template <typename T>
std::vector<int> UFClusterAssigner<T>::operator()(
    const std::vector<T> &densities,
    const std::vector<T> &re_weighted_densities, const std::set<int> &noise_pts,
    const std::vector<std::pair<int, double>> &dep_ptrs,
    const std::set<int> &centers) {
  ParUF<int> UF(densities.size());
  parlay::parallel_for(0, densities.size(), [&](int i) {
    if (centers.find(i) == centers.end() &&
        noise_pts.find(i) == noise_pts.end()) {
      UF.link(i, dep_ptrs[i].first);
    }
  });
  std::vector<int> cluster(densities.size());
  parlay::parallel_for(0, densities.size(),
                       [&](int i) { cluster[i] = UF.find(i); });
  return cluster;
}

} // namespace DPC

template parlay::sequence<Tvec_point<float> *>
DPC::construct_graph<float>(const RawDataset &, ParsedDataset &, const unsigned,
                            const float, const int, const int, Distance *,
                            const GraphType);

// For compute_knn
template std::vector<std::pair<int, double>>
DPC::compute_knn<float>(parlay::sequence<Tvec_point<float> *> &,
                        const RawDataset &, const unsigned, const unsigned,
                        Distance *);

// For compute_dep_ptr
template std::vector<std::pair<int, double>>
DPC::compute_dep_ptr<float>(parlay::sequence<Tvec_point<float> *> &,
                            parlay::sequence<Tvec_point<float>> &,
                            const DatasetKnn &, const RawDataset &,
                            const std::vector<double> &, const std::set<int> &,
                            Distance *, unsigned, int);

// For KthDistanceDensityComputer
// template class DPC::KthDistanceDensityComputer<float>;
// template class DPC::KthDistanceDensityComputer<double>;

// For ThresholdCenterFinder
template class DPC::ThresholdCenterFinder<float>;
template class DPC::ThresholdCenterFinder<double>;

// For ProductCenterFinder
template class DPC::ProductCenterFinder<float>;
template class DPC::ProductCenterFinder<double>;

// For UFClusterAssigner
template class DPC::UFClusterAssigner<float>;
template class DPC::UFClusterAssigner<double>;

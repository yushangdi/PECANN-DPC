#include "bruteforce.h"

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

#include "union_find.h"
#include "utils.h"

namespace DPC {
template <class T>
std::vector<int>
cluster_points(std::vector<T> &densities,
               std::vector<std::pair<uint32_t, double>> &dep_ptrs,
               float density_cutoff, float dist_cutoff,
               float center_density_cutoff) {
  // union_find<int> UF(densities.size());
  ParUF<int> UF(densities.size());
  parlay::parallel_for(0, densities.size(), [&](int i) {
    if (dep_ptrs[i].first != densities.size()) { // the max density point
      if (densities[i] > density_cutoff &&
          (dep_ptrs[i].second <= dist_cutoff ||
           densities[i] < center_density_cutoff)) {
        UF.link(i, dep_ptrs[i].first);
      }
    }
  });
  std::vector<int> cluster(densities.size());
  parlay::parallel_for(0, densities.size(),
                       [&](int i) { cluster[i] = UF.find(i); });
  return cluster;
}

// Compute the dependent points for sorted_points[0:threshold). `data_num` is
// the default value is dependent point does not exist. sorted_points have to be
// sorted in descending density order. Only search [:i] for point i.
template <class T>
void bruteforce_dependent_point(
    const unsigned threshold, const std::size_t data_num,
    const parlay::sequence<unsigned> &sorted_points,
    const parlay::sequence<Tvec_point<T>> &points,
    const std::vector<T> &densities,
    std::vector<std::pair<uint32_t, double>> &dep_ptrs,
    const float density_cutoff, Distance *D, const size_t data_dim) {
  parlay::parallel_for(0, threshold, [&](size_t ii) {
    float m_dist = std::numeric_limits<float>::max();
    size_t id = data_num;
    auto i = sorted_points[ii];
    if (densities[i] > density_cutoff) { // skip noise points
      for (size_t jj = 0; jj < ii; jj++) {
        auto j = sorted_points[jj];
        auto dist = D->distance(points[i].coordinates.begin(),
                                points[j].coordinates.begin(), data_dim);
        if (dist <= m_dist) {
          m_dist = dist;
          id = j;
        }
      }
    }
    dep_ptrs[i] = {id, sqrt(m_dist)};
  });
}

// Compute the dependent points for unfinished_points. `data_num` is the default
// value is dependent point does not exist. search through all points in points
template <class T, class Td, class P>
void bruteforce_dependent_point_all(
    const std::size_t data_num,
    const parlay::sequence<unsigned> &unfinished_points,
    const parlay::sequence<Tvec_point<T>> &points,
    const std::vector<Td> &densities,
    std::vector<std::pair<P, double>> &dep_ptrs, Distance *D,
    const size_t data_dim) {
  parlay::parallel_for(0, unfinished_points.size(), [&](size_t ii) {
    float m_dist = std::numeric_limits<float>::max();
    size_t id = data_num;
    auto i = unfinished_points[ii];
    for (size_t j = 0; j < points.size(); j++) {
      if (densities[j] > densities[i] ||
          (densities[j] == densities[i] && j > i)) {
        auto dist = D->distance(points[i].coordinates.begin(),
                                points[j].coordinates.begin(), data_dim);
        if (dist <= m_dist) {
          m_dist = dist;
          id = j;
        }
      }
    }
    dep_ptrs[i] = {id, sqrt(m_dist)};
  });
}

void dpc_bruteforce(const unsigned K, ParsedDataset points,
                    float density_cutoff, float distance_cutoff,
                    float center_density_cutoff, const std::string &output_path,
                    const std::string &decision_graph_path) {

  Distance *D = new Euclidian_Distance();

  parlay::internal::timer t("DPC");

  std::vector<float> densities(points.size);
  parlay::parallel_for(0, points.size, [&](size_t i) {
    std::vector<float> dists(points.size);
    for (size_t j = 0; j < points.size; j++)
      dists[j] = D->distance(points[i].coordinates.begin(),
                             points[j].coordinates.begin(), points.data_dim);
    std::nth_element(dists.begin(), dists.begin() + K, dists.end());
    densities[i] = 1 / sqrt(dists[K]);
  });

  double density_time = t.next_time();
  report(density_time, "Compute density");

  std::vector<std::pair<uint32_t, double>> dep_ptrs(points.size);
  // auto sorted_points= parlay::sequence<unsigned>::from_function(data_num,
  // [](unsigned i){return i;});
  //  parlay::sort_inplace(sorted_points, [&densities](unsigned i, unsigned j){
  // 	return densities[i] > densities[j]  || (densities[i] == densities[j] &&
  // i > j);
  // });
  // bruteforce_dependent_point(data_num, data_num, sorted_points, points,
  // densities, dep_ptrs, density_cutoff, D, data_dim);
  parlay::parallel_for(0, points.size, [&](size_t i) {
    float m_dist = std::numeric_limits<float>::max();
    size_t id = points.size;
    if (densities[i] > density_cutoff) { // skip noise points
      for (size_t j = 0; j < points.size; j++) {
        if (densities[j] > densities[i] ||
            (densities[j] == densities[i] && j > i)) {
          auto dist =
              D->distance(points[i].coordinates.begin(),
                          points[j].coordinates.begin(), points.data_dim);
          if (dist <= m_dist) {
            m_dist = dist;
            id = j;
          }
        }
      }
    }
    dep_ptrs[i] = {id, sqrt(m_dist)};
  });

  double dependent_time = t.next_time();
  report(dependent_time, "Compute dependent points");

  const auto &cluster = cluster_points(densities, dep_ptrs, density_cutoff,
                                       distance_cutoff, center_density_cutoff);
  double cluster_time = t.next_time();
  report(cluster_time, "Find clusters");
  report(density_time + dependent_time + cluster_time, "Total");

  output(densities, cluster, dep_ptrs, output_path, decision_graph_path);
}
} // namespace DPC

template void DPC::bruteforce_dependent_point_all<float, double, int>(
    const std::size_t data_num,
    const parlay::sequence<unsigned> &unfinished_points,
    const parlay::sequence<Tvec_point<float>> &points,
    const std::vector<double> &densities,
    std::vector<std::pair<int, double>> &dep_ptrs, Distance *D,
    const size_t data_dim);

template void DPC::bruteforce_dependent_point_all<float, float, uint32_t>(
    const std::size_t data_num,
    const parlay::sequence<unsigned> &unfinished_points,
    const parlay::sequence<Tvec_point<float>> &points,
    const std::vector<float> &densities,
    std::vector<std::pair<uint32_t, double>> &dep_ptrs, Distance *D,
    const size_t data_dim);

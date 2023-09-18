#include "dpc_framework.h"

namespace DPC {

template <class T>
parlay::sequence<Tvec_point<T> *>
construct_graph(const RawDataset &raw_data, const unsigned L, const float alpha,
                const int max_degree, const Distance *D,
                const GraphType graph_type) {
  return {};
}

template <class T>
std::vector<std::pair<int, double>>
compute_knn(parlay::sequence<Tvec_point<T> *> &graph,
            const RawDataset &raw_data, const unsigned K, const unsigned L,
            const Distance *D) {
  return {};
}

template <class T>
std::vector<std::pair<int, double>>
compute_knn_bruteforce(const RawDataset &raw_data, const unsigned K,
                       const Distance *D) {
  return {};
}

template <class T>
std::vector<std::pair<int, double>>
compute_dep_ptr(parlay::sequence<Tvec_point<T> *> &graph,
                const RawDataset &raw_data, const std::vector<T> &densities,
                const Distance *D, unsigned L, int round_limit) {
  return {};
}

template <class T>
std::vector<std::pair<int, double>>
compute_dep_ptr_bruteforce(const RawDataset &raw_data,
                           const std::vector<T> &densities,
                           const float density_cutoff, Distance *D) {
  return {};
}

} // namespace DPC

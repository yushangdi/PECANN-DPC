#pragma once

#include "utils.h"

namespace DPC {

void dpc(const unsigned K, const unsigned L, const unsigned Lnn,
         const std::string &data_path, float density_cutoff,
         float distance_cutoff, float center_density_cutoff,
         const std::string &output_path, const std::string &decision_graph_path,
         const unsigned Lbuild, const unsigned max_degree, const float alpha,
         const unsigned num_clusters, Method method, GraphType graph_type);

inline void dpc_temp(const std::string &data_path, const unsigned K,
                     const unsigned L, const unsigned Lnn, float density_cutoff,
                     float distance_cutoff, float center_density_cutoff,
                     const std::string &output_path,
                     const std::string &decision_graph_path,
                     const unsigned Lbuild, const unsigned max_degree,
                     const float alpha, const unsigned num_clusters) {
  dpc(K, L, Lnn, data_path, density_cutoff, distance_cutoff,
      center_density_cutoff, output_path, decision_graph_path, Lbuild,
      max_degree, alpha, num_clusters, Method::Doubling, GraphType::BruteForce);
}
} // namespace DPC
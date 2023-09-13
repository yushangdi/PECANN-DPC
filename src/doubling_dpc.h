#pragma once

#include "IO.h"
#include "utils.h"
#include <sstream>

namespace DPC {

std::unordered_map<std::string, double>
dpc(const unsigned K, const unsigned L, const unsigned Lnn, RawDataset raw_data,
    float density_cutoff, float distance_cutoff, float center_density_cutoff,
    const std::string &output_path, const std::string &decision_graph_path,
    const unsigned Lbuild, const unsigned max_degree, const float alpha,
    const unsigned num_clusters, Method method, GraphType graph_type);

inline std::unordered_map<std::string, double> dpc_filenames_temp(
    const std::string &data_path, const unsigned K, const unsigned L,
    const unsigned Lnn, float density_cutoff, float distance_cutoff,
    float center_density_cutoff, const std::string &output_path,
    const std::string &decision_graph_path, const unsigned Lbuild,
    const unsigned max_degree, const float alpha, const unsigned num_clusters,
    const std::string &method_str, const std::string &graph_type_str) {

  Method method;
  std::istringstream(method_str) >> method;
  GraphType graph_type;
  std::istringstream(graph_type_str) >> graph_type;

  RawDataset raw_data(data_path);

  auto result =
      dpc(K, L, Lnn, data_path, density_cutoff, distance_cutoff,
          center_density_cutoff, output_path, decision_graph_path, Lbuild,
          max_degree, alpha, num_clusters, method, graph_type);

  aligned_free(raw_data.data);

  return result;
}
} // namespace DPC
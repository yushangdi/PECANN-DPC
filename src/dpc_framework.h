#pragma once

#include <cstring>

#include "IO.h"
#include "computers.h"
#include "dpc_components.h"
#include "utils.h"

namespace DPC {

struct ClusteringResult {
  std::unordered_map<std::string, double> output_metadata;
  std::vector<int> clusters;
};

ClusteringResult dpc_framework(
    const unsigned K, const unsigned L, const unsigned Lnn, RawDataset raw_data,
    const std::shared_ptr<CenterFinder<double>> &center_finder,
    std::unique_ptr<DPC::DensityComputer> &density_computer),
    const std::string &output_path, const std::string &decision_graph_path,
    const unsigned Lbuild, const unsigned max_degree, const float alpha,
    const unsigned num_clusters, Method method, GraphType graph_type);

} // namespace DPC
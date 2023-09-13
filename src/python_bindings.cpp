#include "doubling_dpc.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(dpc_ann_ext, m) {
  m.def("dpc", &DPC::dpc_temp, "data_path"_a, "K"_a = 6, "L"_a = 12,
        "Lnn"_a = 4, "density_cutoff"_a = 0,
        "distance_cutoff"_a = std::numeric_limits<float>::max(),
        "center_density_cutoff"_a = 0, "output_path"_a = "",
        "decision_graph_path"_a = "", "Lbuild"_a = 12, "max_degree"_a = 16,
        "alpha"_a = 1.2, "num_clusters"_a = 4, "method"_a = "Doubling",
        "graph_type"_a = "Vamana",
        "This function clusters the passed in files.");
}
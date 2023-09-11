#include "doubling_dpc.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

using namespace nb::literals;

// TODO: Accept method type string
NB_MODULE(dpc_ann_ext, m) {
  m.def("dpc", &DPC::dpc_temp, "data_path"_a, "K"_a = 6, "L"_a = 12,
        "Lnn"_a = 4, "density_cutoff"_a = 0, "distance_cutoff"_a = 0,
        "center_density_cutoff"_a = std::numeric_limits<float>::max(),
        "output_path"_a = "", "decision_graph_path"_a = "", "Lbuild"_a = 12,
        "max_degree"_a = 16, "alpha"_a = 1.2, "num_clusters"_a = 4,
        "This function clusters the passed in files.");
}
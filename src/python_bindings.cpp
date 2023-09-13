#include "IO.h"
#include "doubling_dpc.h"
#include "utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <sstream>

namespace nb = nanobind;

using namespace nb::literals;

// TODO(Josh): Make data requirment more flexible? e.g. allow doubles or ints
std::unordered_map<std::string, double>
dpc_numpy(nb::ndarray<float, nb::shape<nb::any, nb::any>, nb::device::cpu,
                      nb::c_contig>
              data,
          const unsigned K, const unsigned L, const unsigned Lnn,
          float density_cutoff, float distance_cutoff,
          float center_density_cutoff, const std::string &output_path,
          const std::string &decision_graph_path, const unsigned Lbuild,
          const unsigned max_degree, const float alpha,
          const unsigned num_clusters, const std::string &method_str,
          const std::string &graph_type_str) {

  float *data_ptr = data.data();
  size_t num_data = data.shape(0);
  size_t data_dim = data.shape(1);
  RawDataset raw_data(data_ptr, num_data, data_dim);

  DPC::Method method;
  std::istringstream(method_str) >> method;
  DPC::GraphType graph_type;
  std::istringstream(graph_type_str) >> graph_type;

  return DPC::dpc(K, L, Lnn, raw_data, density_cutoff, distance_cutoff,
                  center_density_cutoff, output_path, decision_graph_path,
                  Lbuild, max_degree, alpha, num_clusters, method, graph_type);
}

// TODO(Josh): Have code optionally return the clustering instead of writing to
// a file
NB_MODULE(dpc_ann_ext, m) {
  m.def("dpc_filenames", &DPC::dpc_filenames_temp, "data_path"_a, "K"_a = 6,
        "L"_a = 12, "Lnn"_a = 4, "density_cutoff"_a = 0,
        "distance_cutoff"_a = std::numeric_limits<float>::max(),
        "center_density_cutoff"_a = 0,
        "output_path"_a = "", "decision_graph_path"_a = "", "Lbuild"_a = 12,
        "max_degree"_a = 16, "alpha"_a = 1.2, "num_clusters"_a = 4,
        "method"_a = "Doubling", "graph_type"_a = "Vamana",
        "This function clusters the passed in files.");

  // Don't want to allow conversion since then it will copy
  m.def("dpc_numpy", &dpc_numpy, "data"_a.noconvert(), "K"_a = 6, "L"_a = 12,
        "Lnn"_a = 4, "density_cutoff"_a = 0,
        "distance_cutoff"_a = std::numeric_limits<float>::max(),
        "center_density_cutoff"_a = 0,
        "output_path"_a = "", "decision_graph_path"_a = "", "Lbuild"_a = 12,
        "max_degree"_a = 16, "alpha"_a = 1.2, "num_clusters"_a = 4,
        "method"_a = "Doubling", "graph_type"_a = "Vamana",
        "This function clusters the passed in files.");
}

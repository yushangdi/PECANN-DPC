#include "IO.h"
#include "dpc_components.h"
#include "dpc_framework.h"
#include "utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <sstream>

bool report_stats = true;

namespace nb = nanobind;

using namespace nb::literals;

// TODO(Josh): Make data requirment more flexible? e.g. allow doubles or ints
DPC::ClusteringResult
dpc_numpy(nb::ndarray<float, nb::shape<nb::any, nb::any>, nb::device::cpu,
                      nb::c_contig>
              data,
          const unsigned K, const unsigned L, const unsigned Lnn,
          std::shared_ptr<DPC::CenterFinder<double>> center_finder,
          std::shared_ptr<DPC::DensityComputer> density_computer,
          const std::string &output_path,
          const std::string &decision_graph_path, const unsigned Lbuild,
          const unsigned max_degree, const float alpha,
          const unsigned num_clusters, const std::string &method_str,
          const std::string &graph_type_str) {

  // For some reason default arguments in the nanobind call below aren't
  // working, so need to check for null and set here
  if (!center_finder) {
    center_finder = std::make_shared<DPC::ThresholdCenterFinder<double>>();
  }
  if (!density_computer) {
    density_computer = std::make_shared<DPC::KthDistanceDensityComputer>();
  }

  float *data_ptr = data.data();
  size_t num_data = data.shape(0);
  size_t data_dim = data.shape(1);
  if (data_dim % 8 != 0) {
    throw std::invalid_argument(
        "For now, we only support data_dim that are a multiple of 8 for numpy "
        "data to avoid copies, so please pad your data if this is no the "
        "case.");
  }
  RawDataset raw_data(data_ptr, num_data, data_dim);

  DPC::Method method;
  std::istringstream(method_str) >> method;
  DPC::GraphType graph_type;
  std::istringstream(graph_type_str) >> graph_type;

  return DPC::dpc_framework(K, L, Lnn, raw_data, center_finder,
                            density_computer, output_path, decision_graph_path,
                            Lbuild, max_degree, alpha, num_clusters, method,
                            graph_type);
}

DPC::ClusteringResult
dpc_filenames(const std::string &data_path, const unsigned K, const unsigned L,
              const unsigned Lnn,
              std::shared_ptr<DPC::CenterFinder<double>> center_finder,
              std::shared_ptr<DPC::DensityComputer> density_computer,
              const std::string &output_path,
              const std::string &decision_graph_path, const unsigned Lbuild,
              const unsigned max_degree, const float alpha,
              const unsigned num_clusters, const std::string &method_str,
              const std::string &graph_type_str) {

  if (!center_finder) {
    center_finder = std::make_shared<DPC::ThresholdCenterFinder<double>>();
  }
  if (!density_computer) {
    density_computer = std::make_shared<DPC::KthDistanceDensityComputer>();
  }

  DPC::Method method;
  std::istringstream(method_str) >> method;
  DPC::GraphType graph_type;
  std::istringstream(graph_type_str) >> graph_type;

  RawDataset raw_data(data_path);

  DPC::ClusteringResult result =
      DPC::dpc_framework(K, L, Lnn, raw_data, center_finder, density_computer,
                         output_path, decision_graph_path, Lbuild, max_degree,
                         alpha, num_clusters, method, graph_type);

  aligned_free(raw_data.data);

  return result;
}

NB_MODULE(dpc_ann_ext, m) {
  m.def("dpc_filenames", &dpc_filenames, "data_path"_a, "K"_a = 6, "L"_a = 12,
        "Lnn"_a = 4, "center_finder"_a = nullptr,
        "density_computer"_a = nullptr, "output_path"_a = "",
        "decision_graph_path"_a = "", "Lbuild"_a = 12, "max_degree"_a = 16,
        "alpha"_a = 1.2, "num_clusters"_a = 4, "method"_a = "Doubling",
        "graph_type"_a = "Vamana",
        "This function clusters the passed in numpy data and returns a "
        "ClusteringResult object with the clusters and metadata about the "
        "clustering process (including fine grained timing results).");

  // Don't want to allow conversion since then it will copy
  m.def("dpc_numpy", &dpc_numpy, "data"_a.noconvert(), "K"_a = 6, "L"_a = 12,
        "Lnn"_a = 4, "center_finder"_a = nullptr,
        "density_computer"_a = nullptr, "output_path"_a = "",
        "decision_graph_path"_a = "", "Lbuild"_a = 12, "max_degree"_a = 16,
        "alpha"_a = 1.2, "num_clusters"_a = 4, "method"_a = "Doubling",
        "graph_type"_a = "Vamana",
        "This function clusters the passed in files and returns a "
        "ClusteringResult object with the clusters and metadata about the "
        "clustering process (including fine grained timing results).");

  nb::class_<DPC::ClusteringResult>(m, "ClusteringResult")
      .def_rw("clusters", &DPC::ClusteringResult::clusters)
      .def_rw("metadata", &DPC::ClusteringResult::output_metadata);

  nb::class_<DPC::CenterFinder<double>>(m, "CenterFinder");

  nb::class_<DPC::ThresholdCenterFinder<double>, DPC::CenterFinder<double>>(
      m, "ThresholdCenterFinder")
      .def(nb::init<double, double>(),
           "dependant_dist_threshold"_a = std::numeric_limits<float>::max(),
           "density_threshold"_a = 0);

  nb::class_<DPC::ProductCenterFinder<double>, DPC::CenterFinder<double>>(
      m, "ProductCenterFinder")
      .def(nb::init<int, bool>(), "num_clusters"_a, "use_reweighted_density"_a);

  nb::class_<DPC::DensityComputer>(m, "DensityComputer");

  nb::class_<DPC::KthDistanceDensityComputer, DPC::DensityComputer>(
      m, "KthDistanceDensityComputer")
      .def(nb::init());

  nb::class_<DPC::NormalizedDensityComputer, DPC::DensityComputer>(
      m, "NormalizedDensityComputer")
      .def(nb::init());

  nb::class_<DPC::ExpSquaredDensityComputer, DPC::DensityComputer>(
      m, "ExpSquaredDensityComputer")
      .def(nb::init());

  nb::class_<DPC::MutualKNNDensityComputer, DPC::DensityComputer>(
      m, "MutualKNNDensityDensityComputer")
      .def(nb::init());
}

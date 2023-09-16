#pragma once

#include "dpc_framework.h"

namespace DPC {

template <typename T>
class KthDistanceDensityComputer : public DensityComputer<T> {
public:
  // Here we're passing the necessary arguments to the base class constructor
  KthDistanceDensityComputer(const DatasetKnn &data_knn)
      : DPCComputer(data_knn) {}

  // Return the density.
  std::vector<double>
  operator()(parlay::sequence<Tvec_point<T> *> &graph) override;

  // Reweight the density of each point in $v$ based on knn.
  std::vector<double>
  reweight_density(const std::vector<double> &densities) override;
};

} // namespace DPC
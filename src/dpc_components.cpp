#include "dpc_components.h"

namespace DPC {

template <typename T>
std::vector<double> KthDistanceDensityComputer<T>::operator()(
    parlay::sequence<Tvec_point<T> *> &graph) {
  int data_num = this->data_num_;
  int k = this->k_;
  std::vector<double> densities(data_num);
  parlay::parallel_for(0, data_num, [&](int i) {
    densities[i] = 1.0 / (this->knn_[(i + 1) * k - 1].second);
  });
  return densities;
}

template <typename T>
std::vector<double> KthDistanceDensityComputer<T>::reweight_density(
    const std::vector<double> &densities) {
  return {};
}

} // namespace DPC
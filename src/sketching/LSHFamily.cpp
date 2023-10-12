#include "LSHFamily.h"
#include <random>

namespace DPC::Sketching {

void CosineFamily::init(size_t num_hash_functions, size_t data_dim) {
  data_dim_ = data_dim;
  std::default_random_engine generator(seed_);
  std::normal_distribution<float> distribution(0.0, 1.0);
  random_normals = std::vector<float>(num_hash_functions * data_dim);
  for (size_t i = 0; i < random_normals.size(); i++) {
    random_normals.at(i) = distribution(generator);
  }
}

std::vector<size_t> CosineFamily::hash(const float *data) {
  size_t num_outputs = random_normals.size() / data_dim_; 
  auto result = std::vector<size_t>(num_outputs);
  for (size_t i = 0; i < num_outputs; i ++) {
    float total = 0;
    for (size_t j = 0; j < data_dim_; j++) {
      total += random_normals.at(i * data_dim_ + j) * data[j];
    }
    result.at(i) = total > 0;
  }
  return result;
}

}
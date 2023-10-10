#include "RACE.h"
#include <cmath>

RACE::RACE(size_t num_estimators, size_t hashes_per_estimator, size_t data_dim,
           std::shared_ptr<LSHFamily> lsh_family)
    : data_dim_(data_dim), num_estimators_(num_estimators),
      hashes_per_estimator_(hashes_per_estimator), lsh_funcs_(lsh_family),
      race_array(num_estimators *
                 std::pow(lsh_funcs_->hash_range(), hashes_per_estimator)) {
  lsh_funcs_->init(num_estimators * hashes_per_estimator, data_dim_);
  for (size_t i = 0; i < race_array.size(); i++) {
    race_array.at(i) = 0;
  }
}

void RACE::add(const float *data) {
  auto hashes = get_hashes(data);
  for (size_t i = 0; i < num_estimators_; i++) {
    size_t hash = hashes.at(i);
    race_array.at(i * num_estimators_ + hash)++;
  }
}

size_t RACE::query(const float *data) {
  auto hashes = get_hashes(data);
  size_t total = 0;
  for (size_t i = 0; i < num_estimators_; i++) {
    size_t hash = hashes.at(i);
    total += race_array[i * num_estimators_ + hash];
  }
  return total;
}


std::vector<size_t> RACE::get_hashes(const float *data) {
  auto uncombined_hashes = lsh_funcs_->hash(data);
  auto result = std::vector<size_t>(num_estimators_);
  for (size_t i = 0; i < num_estimators_; i++) {
    size_t total = 0;
    for (size_t j = 0; j < hashes_per_estimator_; j++) {
      total += uncombined_hashes.at(i * hashes_per_estimator_ + j);
      total *= lsh_funcs_->hash_range();
    }
    result.at(i) = total;
  }
  return uncombined_hashes;
}


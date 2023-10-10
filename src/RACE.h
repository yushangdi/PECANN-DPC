#include <atomic>
#include <memory>
#include <random>
#include <vector>

// TODO(Josh): Replace float with template
// TODO(Josh): Benchmark with and without std::atomic
// TODO(Josh): Move impls

class LSHFamily {
public:
  virtual void init(size_t num_hash_functions, size_t data_dim) = 0;

  virtual std::vector<size_t> hash(const float *data) = 0;

  virtual size_t hash_range() = 0;
};

class CosineFamily : public LSHFamily {
public:
  CosineFamily(size_t seed) : seed_(seed) {}

  void init(size_t num_hash_functions, size_t data_dim) override {
    data_dim_ = data_dim;
    std::default_random_engine generator(seed_);
    std::normal_distribution<float> distribution(0.0, 1.0);
    random_normals = std::vector<float>(num_hash_functions * data_dim);
    for (size_t i = 0; i < random_normals.size(); i++) {
      random_normals.at(i) = distribution(generator);
    }
  }

  std::vector<size_t> hash(const float *data) override {
    auto result = std::vector<size_t>(random_normals.size() / data_dim_);
    for (size_t i = 0; i < random_normals.size(); i += data_dim_) {
      float total = 0;
      for (size_t j = 0; j < data_dim_; j++) {
        total += random_normals[i * data_dim_ + j] * data[j];
      }
      result.at(i) = total > 0;
    }
    return result;
  }

  size_t hash_range() override { return 2; }

private:
  size_t data_dim_ = 0;
  size_t seed_;
  std::vector<float> random_normals;
};

class RACE {
public:
  RACE(size_t num_estimators, size_t hashes_per_estimator, size_t data_dim,
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

  void add(const float *data) {
    auto hashes = get_hashes(data);
    for (size_t i = 0; i < num_estimators_; i++) {
      size_t hash = hashes.at(i);
      race_array[i * num_estimators_ + hash]++;
    }
  }

  size_t query(const float *data) {
    auto hashes = get_hashes(data);
    size_t total = 0;
    for (size_t i = 0; i < num_estimators_; i++) {
      size_t hash = hashes.at(i);
      total += race_array[i * num_estimators_ + hash];
    }
    return total;
  }

private:
  std::vector<size_t> get_hashes(const float *data) {
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

  size_t num_estimators_, hashes_per_estimator_, data_dim_;
  std::shared_ptr<LSHFamily> lsh_funcs_;
  std::vector<std::atomic<size_t>> race_array;
};
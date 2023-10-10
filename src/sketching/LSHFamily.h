#include <cstddef>
#include <vector>

class LSHFamily {
public:
  virtual void init(size_t num_hash_functions, size_t data_dim) = 0;

  virtual std::vector<size_t> hash(const float *data) = 0;

  virtual size_t hash_range() = 0;
};

class CosineFamily : public LSHFamily {
public:
  CosineFamily(size_t seed) : seed_(seed) {}

  void init(size_t num_hash_functions, size_t data_dim) override;

  std::vector<size_t> hash(const float *data) override;

  size_t hash_range() override { return 2; }

private:
  size_t data_dim_ = 0;
  size_t seed_;
  std::vector<float> random_normals;
};
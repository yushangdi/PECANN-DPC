#include <atomic>
#include <memory>
#include <vector>
#include "LSHFamily.h"

// TODO(Josh): Replace float with template
// TODO(Josh): Benchmark with and without std::atomic
// Add better tests

class RACE {
public:
  RACE(size_t num_estimators, size_t hashes_per_estimator, size_t data_dim,
       std::shared_ptr<LSHFamily> lsh_family);

  void add(const float *data);

  double query(const float *data);

  RACE (const RACE&) = delete;
  RACE& operator= (const RACE&) = delete;

private:
  std::vector<size_t> get_hashes(const float *data);

  size_t num_estimators_, hashes_per_estimator_, data_dim_;
  std::shared_ptr<LSHFamily> lsh_funcs_;
  std::vector<std::atomic<size_t>> race_array;
};
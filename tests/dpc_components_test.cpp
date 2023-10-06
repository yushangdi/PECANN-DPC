bool report_stats = false;

#include "src/dpc_components.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace DPC {

template <class T> void print_vec(T &vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << vec[i];
    if (i != vec.size() - 1) {
      std::cout << ", ";
    }
  }
}

float *get_aligned_data(std::vector<float> &data_vec, int data_dim,
                        int &rounded_dim) {
  float *data;
  rounded_dim = ROUND_UP(data_dim, 8);
  size_t allocSize = data_vec.size() * rounded_dim * sizeof(float);
  alloc_aligned(((void **)&data), allocSize, 8 * sizeof(float));
  for (size_t i = 0; i < data_vec.size(); i++) {
    for (size_t d = 0; d < data_dim; d++) {
      *(data + i * rounded_dim + d) = data_vec[i * data_dim + d];
    }
    memset(data + i * rounded_dim + data_dim, 0,
           (rounded_dim - data_dim) * sizeof(float));
  }
  return data;
}

class SmallDPCFrameworkTest : public ::testing::Test {
protected:
  Distance *D;
  float *data;
  size_t num_data = 10;
  size_t data_dim = 2;
  size_t aligned_dim;
  std::vector<std::pair<int, double>> knn_expected{
      {0, 0.0}, {1, 5.0}, {2, 20.0}, {1, 0.0}, {0, 5.0}, {2, 5.0},
      {2, 0.0}, {1, 5.0}, {3, 5.0},  {3, 0.0}, {2, 5.0}, {4, 5.0},
      {4, 0.0}, {3, 5.0}, {5, 5.0},  {5, 0.0}, {4, 5.0}, {6, 5.0},
      {6, 0.0}, {5, 5.0}, {7, 5.0},  {7, 0.0}, {6, 5.0}, {8, 5.0},
      {8, 0.0}, {7, 5.0}, {9, 5.0},  {9, 0.0}, {8, 5.0}, {7, 20.0}};

  void SetUp() override {
    D = new Euclidian_Distance();
    // y = 2x for x in 1 -- 10.
    std::vector<float> data_vec = {1, 2,  2, 4,  3, 6,  4, 8,  5,  10,
                                   6, 12, 7, 14, 8, 16, 9, 18, 10, 20};
    int rounded_dim;
    data = get_aligned_data(data_vec, data_dim, rounded_dim);
    aligned_dim = rounded_dim;
  }

  void TearDown() override {
    delete D;
    free(data);
  }
};

TEST_F(SmallDPCFrameworkTest, DistTest) {
  auto d = D->distance(data, data + aligned_dim, data_dim);
  EXPECT_EQ(d, 5);

  std::vector<uint8_t> a{1, 2};
  std::vector<uint8_t> b{2, 4};
  d = D->distance(&a[0], &b[0], data_dim);
  EXPECT_EQ(d, 5);
}

TEST_F(SmallDPCFrameworkTest, ConstructGraphTest) {
  using T = float;
  ParsedDataset parsed_data;
  int Lbuild = 3;
  int alpha = 1.2;
  int max_degree = 2;
  int num_clusters = 1;
  GraphType graph_type = GraphType::Vamana;
  RawDataset raw_data = RawDataset(data, num_data, data_dim, aligned_dim);
  auto graph = construct_graph<T>(raw_data, parsed_data, Lbuild, alpha,
                                  max_degree, num_clusters, D, graph_type);
  auto d = D->distance(graph[0]->coordinates.begin(),
                       graph[1]->coordinates.begin(), data_dim);
  EXPECT_EQ(d, 5);
  EXPECT_EQ(graph.size(), num_data);
  EXPECT_EQ(parsed_data.size, num_data);
  EXPECT_EQ(parsed_data.points.size(), num_data);
  EXPECT_EQ(parsed_data.points[1].id, 1);
}

void check_knn(std::vector<std::pair<int, double>> &knn,
               const std::vector<std::pair<int, double>> &expected, int K,
               int num_data) {
  for (int i = 0; i < num_data; ++i) {
    for (int j = 0; j < K; ++j) {
      auto id = i * K + j;
      EXPECT_EQ(knn[id].first, expected[id].first)
          << "Mismatch at point " << i << ", k-NN index " << j
          << " for first element.";

      EXPECT_EQ(knn[id].second, expected[id].second)
          << "Mismatch at point " << i << ", k-NN index " << j
          << " for second element.";
    }
  }
}

TEST_F(SmallDPCFrameworkTest, KNNTest) {
  using T = float;
  ParsedDataset parsed_data;
  int Lbuild = 10;
  int alpha = 1.2;
  int max_degree = 10;
  int num_clusters = 1;
  GraphType graph_type = GraphType::Vamana;
  RawDataset raw_data = RawDataset(data, num_data, data_dim, aligned_dim);
  auto graph = construct_graph<T>(raw_data, parsed_data, Lbuild, alpha,
                                  max_degree, num_clusters, D, graph_type);
  int Lnn = 8;
  int K = 3;
  auto knn = compute_knn(graph, raw_data, K, Lnn, D);
  EXPECT_EQ(knn.size(), K * num_data);
  check_knn(knn, knn_expected, K, num_data);

  knn = compute_knn_bruteforce(raw_data, K, D);
  EXPECT_EQ(knn.size(), K * num_data);
  check_knn(knn, knn_expected, K, num_data);
}

TEST_F(SmallDPCFrameworkTest, DepPtrTest) {
  RawDataset raw_data = RawDataset(data, num_data, data_dim, aligned_dim);
  int K = 3;
  DatasetKnn data_knn(raw_data, D, K, knn_expected);
  std::set<int> noise_pts{1};
  std::vector<double> densities{5, 1, 2, 3, 2, 3, 2, 3, 2, 3};
  auto dep_ptrs =
      compute_dep_ptr_bruteforce(raw_data, data_knn, densities, noise_pts, D);
  EXPECT_EQ(dep_ptrs.size(), num_data);
  const double max_dist = sqrt(std::numeric_limits<double>::max());
  EXPECT_THAT(dep_ptrs[0], Pair(num_data, max_dist));    // max density
  EXPECT_THAT(dep_ptrs[1], Pair(num_data, max_dist));    // noise data
  EXPECT_THAT(dep_ptrs[2], Pair(3, sqrt(5)));            // within knn
  EXPECT_THAT(dep_ptrs[3], Pair(5, sqrt(20)));           // not within knn
  EXPECT_THAT(dep_ptrs[9], Pair(0, sqrt(81 + 18 * 18))); // not within knn

  using T = float;
  ParsedDataset parsed_data;
  int Lbuild = 10;
  int L = 5;
  int alpha = 1.2;
  int max_degree = 10;
  int num_clusters = 1;
  GraphType graph_type = GraphType::Vamana;
  auto graph = construct_graph<T>(raw_data, parsed_data, Lbuild, alpha,
                                  max_degree, num_clusters, D, graph_type);
  int round_limit = 4;
  dep_ptrs = compute_dep_ptr(graph, parsed_data.points, data_knn, raw_data,
                             densities, noise_pts, D, L, round_limit);
  EXPECT_EQ(dep_ptrs.size(), num_data);
  EXPECT_THAT(dep_ptrs[0], Pair(num_data, max_dist)); // max density
  EXPECT_THAT(dep_ptrs[1], Pair(num_data, max_dist)); // noise data
  EXPECT_THAT(dep_ptrs[2], Pair(3, sqrt(5)));         // within knn
  double tolerance =
      1e-5; // Adjust this value based on your precision requirements
  EXPECT_THAT(dep_ptrs[3], Pair(5, DoubleNear(sqrt(20), tolerance)));
  EXPECT_THAT(dep_ptrs[9], Pair(0, DoubleNear(sqrt(81 + 18 * 18), tolerance)));
}

TEST_F(SmallDPCFrameworkTest, KthDistanceDensityComputerTest) {
  RawDataset raw_data = RawDataset(data, num_data, data_dim, aligned_dim);
  int K = 3;
  DatasetKnn dataset_knn(raw_data, D, K, knn_expected);
  KthDistanceDensityComputer density_computer;
  density_computer.initialize(dataset_knn);
  auto densities = density_computer();

  std::vector<double> expected(num_data);
  expected[0] = 1 / sqrt(20);
  expected[num_data - 1] = 1 / sqrt(20);
  for (int i = 1; i < num_data - 1; ++i) {
    expected[i] = 1 / sqrt(5);
  }

  for (int i = 0; i < num_data; ++i) {
    EXPECT_DOUBLE_EQ(densities[i], expected[i]) << "Mismatch at point " << i;
  }
}

TEST_F(SmallDPCFrameworkTest, NormalizedDensityComputerTest) {
  RawDataset raw_data = RawDataset(data, num_data, data_dim, aligned_dim);
  int K = 3;
  DatasetKnn dataset_knn(raw_data, D, K, knn_expected);
  NormalizedDensityComputer density_computer;
  density_computer.initialize(dataset_knn);
  auto densities = density_computer();

  std::vector<double> expected(num_data);
  expected[0] = 1 / sqrt(20);
  expected[num_data - 1] = 1 / sqrt(20);
  for (int i = 1; i < num_data - 1; ++i) {
    expected[i] = 1 / sqrt(5);
  }

  for (int i = 0; i < num_data; ++i) {
    EXPECT_DOUBLE_EQ(densities[i], expected[i]) << "Mismatch at point " << i;
  }

  std::vector<double> test_densities{5, 1, 2, 3, 2, 3, 2, 3, 2, 13};
  auto new_densities = density_computer.reweight_density(test_densities);
  expected[0] = 5 / (8.0 / 3);
  expected[1] = 1 / (8.0 / 3);
  expected[2] = 2 / (2);
  expected[3] = 3 / (7.0 / 3);
  expected[4] = 2 / (8.0 / 3);
  expected[5] = 3 / (7.0 / 3);
  expected[6] = 2 / (8.0 / 3);
  expected[7] = 3 / (7.0 / 3);
  expected[8] = 2 / (18.0 / 3);
  expected[9] = 13 / (18.0 / 3);

  for (int i = 0; i < num_data; ++i) {
    EXPECT_DOUBLE_EQ(new_densities[i], expected[i])
        << "Mismatch at point " << i;
  }
}

TEST_F(SmallDPCFrameworkTest, ThresholdCenterFinderTest) {
  double distance_cutoff = 10;
  double center_density_cutoff = 2;
  ThresholdCenterFinder<double> center_finder(
      distance_cutoff, center_density_cutoff); // sample threshold values
  RawDataset raw_data = RawDataset(data, num_data, data_dim, aligned_dim);
  int K = 3;
  DatasetKnn dataset_knn(raw_data, D, K, knn_expected);
  std::vector<double> densities{5, 1, 2, 3, 2, 3, 2, 3, 2, 3};
  std::vector<double> reweighted_densities;
  std::set<int> noise_pts{2};
  std::vector<std::pair<int, double>> dep_ptrs(num_data);
  const double max_dist = sqrt(std::numeric_limits<double>::max());
  dep_ptrs[0] = {num_data, max_dist};
  for (int i = 1; i < 6; ++i) {
    dep_ptrs[i] = {i - 1, 9 + i};
  }
  for (int i = 6; i < num_data; ++i) {
    dep_ptrs[i] = {i - 1, 8};
  }
  auto centers =
      center_finder(densities, reweighted_densities, noise_pts, dep_ptrs);
  EXPECT_THAT(centers, UnorderedElementsAre(0, 3, 4, 5));
}

// Function to check if a set contains numbers from 0 to 49
bool ContainsNumbers(const std::set<int> &inputSet) {
  for (int i = 0; i < 50; ++i) {
    if (inputSet.find(i) == inputSet.end()) {
      std::cout << i << std::endl;
      return false; // Number not found in the set
    }
  }
  return true; // All numbers from 1 to 50 are found
}

TEST_F(SmallDPCFrameworkTest, ProductCenterFinderTest) {
  int num_clusters = 50;
  int n = 100;
  ProductCenterFinder<double> center_finder(num_clusters);
  std::vector<double> densities(n);
  std::vector<double> reweighted_densities;
  std::set<int> noise_pts;
  std::vector<std::pair<int, double>> dep_ptrs(n);
  const double max_dist = sqrt(std::numeric_limits<double>::max());
  dep_ptrs[0] = {num_data, max_dist};
  for (int i = 1; i < n; ++i) {
    dep_ptrs[i] = {i - 1, 1.5};
  }
  for (int i = 0; i < n; ++i) {
    densities[i] = n - i;
  }
  auto centers =
      center_finder(densities, reweighted_densities, noise_pts, dep_ptrs);
  ASSERT_EQ(num_clusters, centers.size());
  ASSERT_TRUE(ContainsNumbers(centers));
}

// Function to check if a set contains numbers from 50 to 99
bool ContainsNumbers2(const std::set<int> &inputSet) {
  for (int i = 0; i < 50; ++i) {
    if (inputSet.find(99-i) == inputSet.end()) {
      std::cout << i << std::endl;
      return false; // Number not found in the set
    }
  }
  return true;
}

TEST_F(SmallDPCFrameworkTest, ProductCenterFinderUseWeightedTest) {
  int num_clusters = 50;
  int n = 100;
  ProductCenterFinder<double> center_finder(num_clusters, true);
  std::vector<double> densities(n);
  std::vector<double> reweighted_densities(n);
  std::set<int> noise_pts;
  std::vector<std::pair<int, double>> dep_ptrs(n);
  const double max_dist = sqrt(std::numeric_limits<double>::max());
  dep_ptrs[n-1] = {num_data, max_dist};
  for (int i = 0; i < n-1; ++i) {
    dep_ptrs[i] = {i - 1, 1.5};
  }
  for (int i = 0; i < n; ++i) {
    densities[i] = n - i;
    reweighted_densities[i] = i;
  }
  auto centers =
      center_finder(densities, reweighted_densities, noise_pts, dep_ptrs);
  ASSERT_EQ(num_clusters, centers.size());
  ASSERT_TRUE(ContainsNumbers2(centers));
}


// Test case for when use_reweighted_density_ is true
TEST_F(SmallDPCFrameworkTest, UseReweightedDensity) {
    ProductCenterFinder<double> finder(2, true);
    
    std::vector<double> densities = {1.0, 2.0, 300};
    std::vector<double> re_weighted_densities = {6.0, 4.0, 2.0};
    std::set<int> noise_pts = {2};
    std::vector<std::pair<int, double>> dep_ptrs = {{0, 100}, {2, 200}, {0, 0.9}};
    
    // Act
    std::set<int> centers = finder(densities, re_weighted_densities, noise_pts, dep_ptrs);
    
    // Assert
    ASSERT_EQ(centers.size(), 2);
    EXPECT_THAT(centers, UnorderedElementsAre(0, 1));
}

// Test case for when use_reweighted_density_ is false
TEST_F(SmallDPCFrameworkTest, NoReweightedDensity) {
    // Arrange
    ProductCenterFinder<double> finder(2);
    
    std::vector<double> densities = {1.0, 2.0, 3.0};
    std::vector<double> re_weighted_densities;
    std::set<int> noise_pts = {2};
    std::vector<std::pair<int, double>> dep_ptrs = {{0, 100}, {2, 200}, {0, 0.9}};
    
    // Act
    std::set<int> centers = finder(densities, re_weighted_densities, noise_pts, dep_ptrs);
    
    // Assert
    ASSERT_EQ(centers.size(), 2);
    EXPECT_THAT(centers, UnorderedElementsAre(0, 1));
}


TEST_F(SmallDPCFrameworkTest, UFClusterAssignerTest) {
  auto cluster_assigner = UFClusterAssigner<double>();
  RawDataset raw_data = RawDataset(data, num_data, data_dim, aligned_dim);
  int K = 3;
  DatasetKnn dataset_knn(raw_data, D, K, knn_expected);
  cluster_assigner.initialize(dataset_knn);

  std::vector<double> densities{5, 1, 2, 3, 2, 3, 2, 3, 2, 3};
  std::vector<double> reweighted_densities;
  std::set<int> noise_pts{2};
  std::vector<std::pair<int, double>> dep_ptrs(num_data);
  std::set<int> centers({0, 3, 4, 5});
  const double max_dist = sqrt(std::numeric_limits<double>::max());
  dep_ptrs[0] = {num_data, max_dist};
  for (int i = 1; i < 6; ++i) {
    dep_ptrs[i] = {i - 1, 9 + i};
  }
  for (int i = 6; i < num_data; ++i) {
    dep_ptrs[i] = {i - 1, 8};
  }
  auto cluster = cluster_assigner(densities, reweighted_densities, noise_pts,
                                  dep_ptrs, centers);
  // TODO (shangdi): change to test groups instead of actual id
  EXPECT_THAT(cluster, ElementsAre(1, 1, 2, 3, 4, 9, 9, 9, 9, 9));
}

} // namespace DPC

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

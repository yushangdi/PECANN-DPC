bool report_stats = true;

#include "src/dpc_components.h"
#include <gtest/gtest.h>

namespace DPC {

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
               const std::vector<std::vector<std::pair<int, double>>> expected,
               int K, int num_data) {
  for (int i = 0; i < num_data; ++i) {
    for (int j = 0; j < K; ++j) {
      auto id = i * K + j;
      EXPECT_EQ(knn[id].first, expected[i][j].first)
          << "Mismatch at point " << i << ", k-NN index " << j
          << " for first element.";

      EXPECT_EQ(knn[id].second, expected[i][j].second)
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

  const std::vector<std::vector<std::pair<int, double>>> expected = {
      {{0, 0.0}, {1, 5.0}, {2, 20.0}}, {{1, 0.0}, {0, 5.0}, {2, 5.0}},
      {{2, 0.0}, {1, 5.0}, {3, 5.0}},  {{3, 0.0}, {2, 5.0}, {4, 5.0}},
      {{4, 0.0}, {3, 5.0}, {5, 5.0}},  {{5, 0.0}, {4, 5.0}, {6, 5.0}},
      {{6, 0.0}, {5, 5.0}, {7, 5.0}},  {{7, 0.0}, {6, 5.0}, {8, 5.0}},
      {{8, 0.0}, {7, 5.0}, {9, 5.0}},  {{9, 0.0}, {8, 5.0}, {7, 20.0}}};
  check_knn(knn, expected, K, num_data);

  compute_knn_bruteforce(raw_data, K, D);
  EXPECT_EQ(knn.size(), K * num_data);
  check_knn(knn, expected, K, num_data);
}

TEST_F(SmallDPCFrameworkTest, KthDistanceDensityComputerTest) {
  // Test KthDistanceDensityComputer here. This is just a placeholder.
  KthDistanceDensityComputer<double> computer;
  // You'll need to set up necessary inputs and expected outputs.
  // Then invoke methods on `computer` and check results against expected
  // outputs.
}

TEST_F(SmallDPCFrameworkTest, ThresholdCenterFinderTest) {
  // Test ThresholdCenterFinder here. This is just a placeholder.
  ThresholdCenterFinder<double> finder(0.5, 0.5); // sample threshold values
  // You'll need to set up necessary inputs and expected outputs.
  // Then invoke methods on `finder` and check results against expected outputs.
}

TEST_F(SmallDPCFrameworkTest, UFClusterAssignerTest) {
  // Test UFClusterAssigner here. This is just a placeholder.
  UFClusterAssigner<double> assigner;
  // You'll need to set up necessary inputs and expected outputs.
  // Then invoke methods on `assigner` and check results against expected
  // outputs.
}

// Add more TEST_F blocks for other classes and methods as needed.

} // namespace DPC

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

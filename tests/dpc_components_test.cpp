#include "src/dpc_components.h"
#include <gtest/gtest.h>

namespace DPC {

// Sample test cases. You should add more as needed.

class DPCFrameworkTest : public ::testing::Test {
protected:
    Distance *D = new Euclidian_Distance();
};

TEST_F(DPCFrameworkTest, ConstructGraphTest) {
    using T = float;
    std::vector<float> data({1, 1, 0,  2, 2, 0, 3, 3, 0});
    size_t num_data = 3;
    size_t data_dim = 2;
    size_t aligned_dim = 3;
    RawDataset raw_data(data.data(), num_data, data_dim, aligned_dim);
    ParsedDataset parsed_data;
    int Lbuild = 3;
    int alpha = 1.2;
    int max_degree = 2;
    int num_clusters = 1;
    GraphType graph_type = GraphType::Vamana;

    auto graph = construct_graph<T>(raw_data, parsed_data, Lbuild, alpha, max_degree,
                               num_clusters, D, graph_type);
    EXPECT_EQ(graph.size(), num_data);
    EXPECT_EQ(parsed_data.size, num_data);  
    EXPECT_EQ(parsed_data.points.size(), num_data);     
    EXPECT_EQ(parsed_data.points[1].id, 1);      

}

TEST_F(DPCFrameworkTest, KthDistanceDensityComputerTest) {
    // Test KthDistanceDensityComputer here. This is just a placeholder.
    KthDistanceDensityComputer<double> computer;
    // You'll need to set up necessary inputs and expected outputs.
    // Then invoke methods on `computer` and check results against expected outputs.
}

TEST_F(DPCFrameworkTest, ThresholdCenterFinderTest) {
    // Test ThresholdCenterFinder here. This is just a placeholder.
    ThresholdCenterFinder<double> finder(0.5, 0.5); // sample threshold values
    // You'll need to set up necessary inputs and expected outputs.
    // Then invoke methods on `finder` and check results against expected outputs.
}

TEST_F(DPCFrameworkTest, UFClusterAssignerTest) {
    // Test UFClusterAssigner here. This is just a placeholder.
    UFClusterAssigner<double> assigner;
    // You'll need to set up necessary inputs and expected outputs.
    // Then invoke methods on `assigner` and check results against expected outputs.
}

// Add more TEST_F blocks for other classes and methods as needed.

}  // namespace DPC

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

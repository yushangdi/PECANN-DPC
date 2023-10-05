#pragma once

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <string.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "IO.h"

#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

#include "ParlayANN/algorithms/utils/NSGDist.h"

#include "union_find.h"
#include "utils.h"

namespace DPC {
template <class T>
std::vector<int>
cluster_points(std::vector<T> &densities,
               std::vector<std::pair<uint32_t, double>> &dep_ptrs,
               float density_cutoff, float dist_cutoff,
               float center_density_cutoff);

// Compute the dependent points for sorted_points[0:threshold). `data_num` is
// the default value is dependent point does not exist. sorted_points have to be
// sorted in descending density order. Only search [:i] for point i.
template <class T>
void bruteforce_dependent_point(
    const unsigned threshold, const std::size_t data_num,
    const parlay::sequence<unsigned> &sorted_points,
    const parlay::sequence<Tvec_point<T>> &points,
    const std::vector<T> &densities,
    std::vector<std::pair<uint32_t, double>> &dep_ptrs,
    const float density_cutoff, Distance *D, const size_t data_dim);

// Compute the dependent points for unfinished_points. `data_num` is the default
// value is dependent point does not exist. search through all points in points
template <class T, class Td, class P>
void bruteforce_dependent_point_all(
    const std::size_t data_num,
    const parlay::sequence<unsigned> &unfinished_points,
    const parlay::sequence<Tvec_point<T>> &points,
    const std::vector<Td> &densities,
    std::vector<std::pair<P, double>> &dep_ptrs, Distance *D,
    const size_t data_dim);

void dpc_bruteforce(const unsigned K, ParsedDataset points,
                    float density_cutoff, float distance_cutoff,
                    float center_density_cutoff, const std::string &output_path,
                    const std::string &decision_graph_path);
} // namespace DPC
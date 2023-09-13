// This file is temporary, right now we can't include ParlayANN header files in
// two different cc files because it messes up linking since there are functions
// in headers not inlined, once this is fixed we can add this to the general
// utils/io file

#pragma once

#include "IO.h"
#include "ParlayANN/algorithms/utils/types.h"

struct ParsedDataset {
  size_t size;
  size_t data_dim;
  size_t aligned_dim;
  parlay::sequence<Tvec_point<float>> points;
  Tvec_point<float> operator[](size_t i) const { return points[i]; }

  ParsedDataset(RawDataset d)
      : size(d.num_data), data_dim(d.data_dim), aligned_dim(d.aligned_dim) {
    points = parlay::sequence<Tvec_point<float>>(size);
    parlay::parallel_for(0, size, [&](size_t i) {
      float *start = d.data + (i * d.aligned_dim);
      float *end = d.data + ((i + 1) * d.aligned_dim);
      points[i].id = i;
      points[i].coordinates = parlay::make_slice(start, end);
    });
  }
};
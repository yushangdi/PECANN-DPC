#pragma once

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y)                                                         \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)

inline void print_error_and_terminate(std::stringstream &error_stream) {
  std::cerr << error_stream.str() << std::endl;
  throw std::runtime_error(error_stream.str());
}

inline void report_memory_allocation_failure() {
  std::stringstream stream;
  stream << "Memory Allocation Failed.";
  print_error_and_terminate(stream);
}

inline void report_misalignment_of_requested_size(size_t align) {
  std::stringstream stream;
  stream << "Requested memory size is not a multiple of " << align
         << ". Can not be allocated.";
  print_error_and_terminate(stream);
}

inline void alloc_aligned(void **ptr, size_t size, size_t align) {
  *ptr = nullptr;
  if (IS_ALIGNED(size, align) == 0)
    report_misalignment_of_requested_size(align);
#ifndef _WINDOWS
  *ptr = ::aligned_alloc(align, size);
#else
  *ptr = ::_aligned_malloc(size, align); // note the swapped arguments!
#endif
  if (*ptr == nullptr)
    report_memory_allocation_failure();
}

inline void aligned_free(void *ptr) {
  // Gopal. Must have a check here if the pointer was actually allocated by
  // _alloc_aligned
  if (ptr == nullptr) {
    return;
  }
#ifndef _WINDOWS
  free(ptr);
#else
  ::_aligned_free(ptr);
#endif
}

inline bool is_newline(char c) {
  switch (c) {
  case '\r':
  case '\n':
    return true;
  default:
    return false;
  }
}

inline bool is_delim(char c) {
  switch (c) {
  case '\t':
  case ';':
  case ',':
  case ' ':
    return true;
  default:
    return false;
  }
}

inline bool is_space(char c) { return is_newline(c) || is_delim(c) || c == 0; }

inline std::vector<char> readStringFromFile(const std::string &fileName) {
  std::ifstream file(fileName, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file");
  }
  long end = file.tellg();
  file.seekg(0, std::ios::beg);
  long n = end - file.tellg();
  std::vector<char> bytes(n, (char)0);
  file.read(bytes.data(), n);
  file.close();
  return bytes;
}

inline std::pair<std::vector<char *>, size_t>
stringToWords(std::vector<char> &Str) {
  size_t n = Str.size();
  std::vector<char *> SA;
  bool isLastSpace = true;
  size_t dim = 0;
  for (size_t i = 0; i < n; i++) {
    bool isThisSpace = is_space(Str[i]);
    if (!isThisSpace && isLastSpace) {
      SA.push_back(Str.data() + i);
    }
    if (dim == 0 && is_newline(Str[i])) {
      dim = SA.size();
    }
    isLastSpace = isThisSpace;
  }
  return std::make_pair(SA, dim);
}

inline void load_text_file(const std::string &text_file, float *&data,
                           size_t &npts, size_t &dim, size_t &rounded_dim) {
  std::vector<char> chars = readStringFromFile(text_file);
  auto ret = stringToWords(chars);
  std::vector<char *> words(std::move(ret.first));
  dim = ret.second;
  assert(words.size() % dim == 0);
  npts = words.size() / dim;
  rounded_dim = ROUND_UP(dim, 8);
  std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
            << ", aligned_dim = " << rounded_dim << "... " << std::flush;
  size_t allocSize = npts * rounded_dim * sizeof(float);
  std::cout << "allocating aligned memory of " << allocSize << " bytes... "
            << std::flush;
  alloc_aligned(((void **)&data), allocSize, 8 * sizeof(float));
  std::cout << "done. Copying data to mem_aligned buffer..." << std::flush;
  for (size_t i = 0; i < npts; i++) {
    for (size_t d = 0; d < dim; d++) {
      *(data + i * rounded_dim + d) = atof(words[i * dim + d]);
    }
    memset(data + i * rounded_dim + dim, 0,
           (rounded_dim - dim) * sizeof(float));
  }
  std::cout << " done." << std::endl;
}

// This struct does not own the passed in pointer
struct RawDataset {
  size_t num_data;
  size_t data_dim;
  size_t aligned_dim;
  float *data;

  RawDataset(const std::string &data_path) {
    load_text_file(data_path, data, num_data, data_dim, aligned_dim);
    std::cout << "Loaded text file: num_data=" << num_data
              << ", data_dim=" << data_dim << std::endl;
  }

  RawDataset(float *data, size_t num_data, size_t data_dim)
      : num_data(num_data), data_dim(data_dim), aligned_dim(data_dim),
        data(data) {}
};
#pragma once

#include <boost/algorithm/string.hpp> // For boost::to_lower
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace DPC {

inline std::unordered_map<std::string, double> time_reports;

inline void report(double time, std::string str) {
  time_reports[str] = time;
  std::ios::fmtflags cout_settings = std::cout.flags();
  std::cout.precision(4);
  std::cout << std::fixed;
  // std::cout << name << ": ";
  if (str.length() > 0)
    std::cout << str << ": ";
  std::cout << time << std::endl;
  std::cout.flags(cout_settings);
}

enum class Method { Doubling, BlindProbe };

enum class GraphType { Vamana, pyNNDescent, HCNNG, BruteForce };

// Overload the stream insertion operator for the Method enum class
inline std::ostream &operator<<(std::ostream &os, const Method &method) {
  switch (method) {
  case Method::Doubling:
    os << "Doubling";
    break;
  case Method::BlindProbe:
    os << "BlindProbe";
    break;
  default:
    os << "Unknown Method";
    break;
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const GraphType &g) {
  switch (g) {
  case GraphType::Vamana:
    os << "Vamana";
    break;
  case GraphType::pyNNDescent:
    os << "pyNNDescent";
    break;
  case GraphType::HCNNG:
    os << "HCNNG";
    break;
  case GraphType::BruteForce:
    os << "BruteForce";
    break;
  default:
    os << "Unknown Method";
    break;
  }
  return os;
}

inline std::istream &operator>>(std::istream &in, GraphType &type) {
  std::string token;
  in >> token;
  if (token == "Vamana")
    type = GraphType::Vamana;
  else if (token == "pyNNDescent")
    type = GraphType::pyNNDescent;
  else if (token == "HCNNG")
    type = GraphType::HCNNG;
  else if (token == "BruteForce")
    type = GraphType::BruteForce;
  else
    in.setstate(std::ios_base::failbit);
  return in;
}

inline std::istream &operator>>(std::istream &in, Method &method) {
  std::string token;
  in >> token;
  if (token == "Doubling")
    method = Method::Doubling;
  else if (token == "BlindProbe")
    method = Method::BlindProbe;
  else
    in.setstate(std::ios_base::failbit);
  return in;
}

inline void validate(boost::any &v, const std::vector<std::string> &values,
                     Method *, int) {
  namespace po = boost::program_options;

  po::validators::check_first_occurrence(v);

  const std::string &s = po::validators::get_single_string(values);
  std::string lower_s = s;
  boost::to_lower(lower_s);

  if (lower_s == "doubling") {
    v = Method::Doubling;
  } else if (lower_s == "blindprobe") {
    v = Method::BlindProbe;
  } else {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }
}

template <class T>
inline void
output(const std::vector<T> &densities, const std::vector<int> &cluster,
       const std::vector<std::pair<uint32_t, double>> &dep_ptrs,
       const std::string &output_path, const std::string &decision_graph_path) {
  if (output_path != "") {
    std::ofstream fout(output_path);
    for (size_t i = 0; i < cluster.size(); i++) {
      fout << cluster[i] << std::endl;
    }
    fout.close();
  }

  if (decision_graph_path != "") {
    std::cout << "writing decision graph\n";
    std::ofstream fout(decision_graph_path);
    for (size_t i = 0; i < densities.size(); i++) {
      fout << densities[i] << " " << dep_ptrs[i].second << " "
           << dep_ptrs[i].first << '\n';
    }
  }
}

template <typename T>
inline void writeVectorToFile(const std::vector<T> &vec,
                              const std::string &filepath) {
  std::ofstream outFile(filepath);
  if (!outFile) {
    std::cerr << "Error opening file: " << filepath << std::endl;
    return;
  }
  for (const T &item : vec) {
    outFile << item << '\n';
  }
  outFile.close();
}

} // namespace DPC

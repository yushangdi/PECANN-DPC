
int main(int argc, char **argv) {
  using Method = DPC::Method;
  using GraphType = DPC::GraphType;
  std::string query_file, output_file, decision_graph_path;
  float density_cutoff, dist_cutoff, center_density_cutoff;
  bool bruteforce = false;
  unsigned int K = 6;
  unsigned int L = 12;
  unsigned int Lnn = 4;
  unsigned int Lbuild = 12;
  unsigned int max_degree = 16;
  unsigned int num_clusters = 4; // only used for pyNNDescent.
  float alpha = 1.2;
  Method method = Method::Doubling;
  GraphType graph_type = GraphType::Vamana;

  po::options_description desc("DPC");
  desc.add_options()("help", "produce help message")(
      "K", po::value<unsigned int>(&K)->default_value(6),
      "the number of nearest neighbor used for computing the density.")(
      "L", po::value<unsigned int>(&L)->default_value(12),
      "L value used for density computation.")(
      "Lnn", po::value<unsigned int>(&Lnn)->default_value(4),
      "the starting Lnn value used for dependent point computation.")(
      "Lbuild", po::value<unsigned int>(&Lbuild)->default_value(12),
      "Retain closest Lbuild number of nodes during the greedy search of "
      "construction.")("max_degree",
                       po::value<unsigned int>(&max_degree)->default_value(16),
                       "max_degree value used for constructing the graph.")(
      "alpha", po::value<float>(&alpha)->default_value(1.2), "alpha value")(
      "num_clusters", po::value<unsigned int>(&num_clusters)->default_value(4),
      "number of clusters, only used for pyNNDescent graph method.")(
      "query_file", po::value<std::string>(&query_file)->required(),
      "Data set file")("output_file",
                       po::value<std::string>(&output_file)->default_value(""),
                       "Output cluster file")(
      "decision_graph_path",
      po::value<std::string>(&decision_graph_path)->default_value(""),
      "Output decision_graph_path")(
      "density_cutoff", po::value<float>(&density_cutoff)->default_value(0),
      "Density below which points are treated as noise")(
      "center_density_cutoff",
      po::value<float>(&center_density_cutoff)->default_value(0),
      "Density below which points are sorted into the same cluster")(
      "dist_cutoff",
      po::value<float>(&dist_cutoff)
          ->default_value(std::numeric_limits<float>::max()),
      "Distance below which points are sorted into the same cluster")(
      "bruteforce", po::value<bool>(&bruteforce)->default_value(false),
      "Whether bruteforce method is used.")(
      "method", po::value<Method>(&method)->default_value(Method::Doubling),
      "Method (Doubling or BlindProbe). Only works when bruteforce=false.")(
      "graph_type",
      po::value<GraphType>(&graph_type)->default_value(GraphType::Vamana),
      "Graph type (Vamana or pyNNDescent or HCNNG). Only works when "
      "bruteforce=false.")

      ;

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << "\n";
    if (vm.count("help") || argc == 1) {
      std::cout << desc << "\n";
      return 0;
    }
    return 1;
  }

  std::cout << "query_file=" << query_file << "\n";
  std::cout << "output_file=" << output_file << "\n";
  std::cout << "decision_graph_path=" << decision_graph_path << "\n";
  std::cout << "density_cutoff=" << density_cutoff << "\n";
  std::cout << "center_density_cutoff=" << center_density_cutoff << "\n";
  std::cout << "dist_cutoff=" << dist_cutoff << "\n";
  std::cout << "num_thread: " << parlay::num_workers() << std::endl;

  if (bruteforce) {
    std::cout << "method= brute force\n";
    DPC::dpc_bruteforce(K, query_file, density_cutoff, dist_cutoff,
                        center_density_cutoff, output_file,
                        decision_graph_path);
  } else {
    std::cout << "graph_type=" << graph_type << std::endl;
    std::cout << "method=" << method << std::endl;
    std::cout << "K=" << K << "\n";
    std::cout << "L=" << L << "\n";
    std::cout << "Lnn=" << Lnn << "\n";
    std::cout << "Lbuild=" << Lbuild << "\n";
    std::cout << "max_degree=" << max_degree << "\n";
    if (graph_type != GraphType::HCNNG) {
      std::cout << "alpha=" << alpha << "\n";
    }
    if (graph_type == GraphType::pyNNDescent ||
        graph_type == GraphType::HCNNG) {
      std::cout << "num_clusters=" << num_clusters << "\n";
      if (Lbuild < 8) {
        std::cerr << "Please use Lbuild >= 8 for pyNNDescent and HCNNG\n";
        std::cerr << "Lbuild = " << Lbuild << std::endl;
        exit(1);
      }
    }

    DPC::dpc(K, L, Lnn, query_file, density_cutoff, dist_cutoff,
             center_density_cutoff, output_file, decision_graph_path, Lbuild,
             max_degree, alpha, num_clusters, method, graph_type);
  }
}
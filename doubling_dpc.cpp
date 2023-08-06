#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <set>
#include <string.h>
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "IO.h"

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/slice.h"
#include "parlay/internal/get_time.h"

#include "ParlayANN/algorithms/utils/beamSearch.h"
#include "ParlayANN/algorithms/utils/NSGDist.h"

#include "union_find.h"
#include "utils.h"
#include "bruteforce.h"

// g++ -std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG -IParlayANN/parlaylib/include doubling_dpc.cpp -I/home/ubuntu/boost_1_82_0 -o doubling_dpc -lboost_program_options
namespace po = boost::program_options;

bool report_stats = true;



namespace DPC {
// v, i, densities, data_aligned_dim, Lnn, index
template<class T>
std::pair<uint32_t, double> compute_dep_ptr(parlay::sequence<Tvec_point<T>*> data, std::size_t query_id, const std::vector<T>& densities, 
													const size_t data_aligned_dim, unsigned& L, Distance* D, int round_limit = -1){
	// if(L*4 > densities.size()) return densities.size(); // why?
	
	parlay::sequence<Tvec_point<T>*> start_points;
	start_points.push_back(data[query_id]);

	uint32_t dep_ptr;
	float minimum_dist;

	if(round_limit == -1){
		round_limit = densities.size(); // effectively no limit on round.
	}

	for(int round = 0; round < round_limit; ++round){
		auto [pairElts, dist_cmps] = beam_search<T>(data[query_id], data, 
																						start_points, L, data_aligned_dim, D);
		auto [beamElts, visitedElts] = pairElts;

		double query_density = densities[query_id];
		T* query_ptr = data[query_id]->coordinates.begin();
		minimum_dist = std::numeric_limits<float>::max();
		dep_ptr = densities.size();
		for(unsigned i=0; i<beamElts.size(); i++){
			const auto [id, dist] = beamElts[i];
			if (id == query_id) continue;
			// if(id == densities.size()) break;
			if(densities[id] > query_density || (densities[id] == query_density && id > query_id)){
				if(dist < minimum_dist){
					minimum_dist = dist;
					dep_ptr = id;
				}
			}
		}
		if(dep_ptr != densities.size()){
			break;
		}
		L *= 2;
	}

	// if(dep_ptr == densities.size()){
	// 	L *= 2;
	// 	return compute_dep_ptr(data, query_id, densities, data_aligned_dim, L, D);
	// }
	return {dep_ptr, minimum_dist};
}


// v, i, densities, data_aligned_dim, Lnn, index
// template<class T>
// std::pair<uint32_t, double> compute_dep_ptr_blind_probe(parlay::sequence<Tvec_point<T>*> data, std::size_t query_id, const std::vector<T>& densities, 
// 													const size_t data_aligned_dim, unsigned& L, Distance* D){
// 	// if(L*4 > densities.size()) return densities.size(); // why?
	
// 	parlay::sequence<Tvec_point<T>*> start_points;
// 	start_points.push_back(data[query_id]);
// 	auto [pairElts, dist_cmps] = beam_search_blind_probe<T, T>(data[query_id], data, densities,
// 																					start_points, L, data_aligned_dim, D);
// 	auto [beamElts, visitedElts] = pairElts;

// 	double query_density = densities[query_id];
// 	T* query_ptr = data[query_id]->coordinates.begin();
// 	float minimum_dist = std::numeric_limits<float>::max();
// 	uint32_t dep_ptr = densities.size();
// 	for(unsigned i=0; i<beamElts.size(); i++){
// 		const auto [id, dist] = beamElts[i];
// 		if (id == query_id) continue;
// 		// if(id == densities.size()) break;
// 		if(densities[id] > query_density || (densities[id] == query_density && id > query_id)){
// 			if(dist < minimum_dist){
// 				minimum_dist = dist;
// 				dep_ptr = id;
// 			}
// 		} else {
// 			std::cout << "Internal error: blind probe retuned invalid points \n.";
// 		}
// 	}
// 	if(dep_ptr == densities.size()){
// 		L *= 2;
// 		return compute_dep_ptr_blind_probe(data, query_id, densities, data_aligned_dim, L, D);
// 	}
// 	return {dep_ptr, minimum_dist};
// }


template<class T>
void compute_densities(parlay::sequence<Tvec_point<T>*>& v, std::vector<T>& densities, const unsigned L, 
												const unsigned K, const size_t data_num, const size_t data_aligned_dim, Distance* D){
	auto beamSizeQ = L;
	parlay::parallel_for(0, data_num, [&](size_t i) {
    // parlay::sequence<int> neighbors = parlay::sequence<int>(k);
		// what is cut and limit?
		parlay::sequence<Tvec_point<T>*> start_points;
    start_points.push_back(v[i]);
    auto [pairElts, dist_cmps] = beam_search(v[i], v, 
																						start_points, beamSizeQ, data_aligned_dim, D, K);
    auto [beamElts, visitedElts] = pairElts;
		auto less = [&](id_dist a, id_dist b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first); };
		auto sorted_nn = parlay::sort(beamElts, less);
		T distance = sorted_nn[K].second;
		if(distance <= 0){
			densities[i] =  std::numeric_limits<T>::max();
		}else{
			densities[i] =  1.0/distance;
		}
  });
}


void dpc(const unsigned K, const unsigned L, const unsigned Lnn, const std::string& data_path, float density_cutoff, float distance_cutoff,
         const std::string& output_path, const std::string& decision_graph_path, const unsigned Lbuild, const unsigned max_degree, const float alpha, Method method ){
	// using std::chrono::high_resolution_clock;
  // using std::chrono::duration_cast;
  // using std::chrono::duration;
  // using std::chrono::microseconds;
  using T = float;
  parlay::internal::timer t("DPC");

  Distance* D = new Euclidian_Distance();

	T* data = nullptr;
	size_t data_num, data_dim, data_aligned_dim;
	load_text_file(data_path, data, data_num, data_dim,
                               data_aligned_dim);
	//diskann::load_aligned_bin<float>(data_path, data, data_num, data_dim,
      //                         data_aligned_dim);


	std::cout<<"data_num: "<<data_num<<std::endl;

  auto points = parlay::sequence<Tvec_point<T>>(data_num);
  parlay::parallel_for(0, data_num, [&] (size_t i) {
    T* start = data + (i * data_aligned_dim);
    T* end = data + ((i+1) * data_aligned_dim);
    points[i].id = i; 
    points[i].coordinates = parlay::make_slice(start, end);  
  });
	add_null_graph(points, max_degree);
  auto v = parlay::tabulate(data_num, [&] (size_t i) -> Tvec_point<T>* {
      return &points[i];});
  t.next("Load data.");

  using findex = knn_index<T>;
  findex I(max_degree, Lbuild, alpha, data_dim, D);
  parlay::sequence<int> inserts = parlay::tabulate(v.size(), [&] (size_t i){
          return static_cast<int>(i);});
  I.build_index(v, inserts);
	double build_time = t.next_time();
  report(build_time, "Built index");

  if(report_stats){
    auto [avg_deg, max_deg] = graph_stats(v);
    std::cout << "Index built with average degree " << avg_deg << " and max degree " << max_deg << std::endl;
    t.next("stats");
  }

	std::vector<T> densities(data_num);
	compute_densities(v, densities, L, K, data_num, data_aligned_dim, D);
	double density_time = t.next_time();
  report(density_time, "Compute density");

	// sort in desending order
	// auto sorted_points= parlay::sequence<unsigned>::from_function(data_num, [](unsigned i){return i;});
	// parlay::sort_inplace(sorted_points, [&densities](unsigned i, unsigned j){
	// 	return densities[i] > densities[j]  || (densities[i] == densities[j] && i > j);
	// });
	// auto max_point_id = sorted_points[0];
	// unsigned threshold = log(data_num);

	Tvec_point<T>** max_density_point = parlay::max_element(v, [&densities](Tvec_point<T>* a, Tvec_point<T>* b){
		if(densities[a->id] == densities[b->id]){
			return a->id < b->id;
		}
		return densities[a->id] < densities[b->id];
	});
	auto max_point_id = max_density_point[0]->id;
	unsigned threshold = 0;

  std::vector<std::pair<uint32_t, double>> dep_ptrs(data_num);
	// dep_ptrs[max_point_id] = {data_num, -1};
	parlay::parallel_for(0, data_num, [&](size_t i) {dep_ptrs[i] = {data_num, -1};});
	auto unfinished_points= parlay::sequence<unsigned>::from_function(data_num, [](unsigned i){return i;});
	unfinished_points = parlay::filter(unfinished_points, [&](size_t i){
				return i!= max_point_id && densities[i] > density_cutoff; // skip noise points
	});
	//compute the top log n density points using bruteforce
	// std::cout << "threshold: " << threshold << std::endl;
	// bruteforce_dependent_point(0, data_num, sorted_points, points, densities, dep_ptrs, density_cutoff, D, data_dim);
	// bruteforce_dependent_point(threshold, data_num, sorted_points, points, densities, dep_ptrs, density_cutoff, D, data_dim);

	std::vector<unsigned> num_rounds(data_num, Lnn); // the L used when dependent point is found.
	if (method == Method::Doubling){
		int round_limit = 4;
		while(unfinished_points.size() > 300){
			parlay::parallel_for(0, unfinished_points.size(), [&](size_t j) {
				// auto i = sorted_points[j];
				auto i = unfinished_points[j];
					dep_ptrs[i] = compute_dep_ptr(v, i, densities, data_aligned_dim, num_rounds[i], D, round_limit);
				// }
			});
			unfinished_points = parlay::filter(unfinished_points, [&dep_ptrs, &data_num](size_t i){
				return dep_ptrs[i].first == data_num;
			});
			std::cout << "number: " << unfinished_points.size() << std::endl;
		}
		std::cout << "bruteforce number: " << unfinished_points.size() << std::endl;
		bruteforce_dependent_point_all(data_num, unfinished_points, points, densities, dep_ptrs, D, data_dim);
	// } else if (method == Method::BlindProbe){
	// 	parlay::parallel_for(threshold, data_num, [&](size_t j) {
	// 		// auto i = sorted_points[j];
	// 		auto i = unfinished_points[j];
	// 		if (i != max_point_id && densities[i] > density_cutoff){ // skip noise points
	// 			unsigned Li = Lnn;
	// 			dep_ptrs[i] = compute_dep_ptr_blind_probe(v, i, densities, data_aligned_dim, Li, D);
	// 			num_rounds[i] = Li;
	// 		}
	// 	});
	} else {
		std::cout << "Error: method not implemented " << std::endl;
		exit(1);
	}
  aligned_free(data);
	double dependent_time = t.next_time();
	report(dependent_time, "Compute dependent points");

	auto cluster = cluster_points(densities, dep_ptrs, density_cutoff, distance_cutoff);
	double cluster_time = t.next_time();
	report(cluster_time, "Find clusters");
	report(build_time + density_time + dependent_time + cluster_time, "Total");

	output(densities, cluster, dep_ptrs, output_path, decision_graph_path);
	writeVectorToFile(num_rounds, "results/num_rounds.txt");
}

}

int main(int argc, char** argv){
	using Method = DPC::Method;
	std::string query_file, output_file, decision_graph_path;
	float density_cutoff, dist_cutoff;
	bool bruteforce = false;
  unsigned int K = 6;
  unsigned int L = 12;
  unsigned int Lnn = 4;
  unsigned int Lbuild = 12; 
  unsigned int max_degree = 16;
  float alpha = 1.2;
	Method method = Method::Doubling;

	po::options_description desc("DPC");
    desc.add_options()
				("help", "produce help message")
        ("K", po::value<unsigned int>(&K)->default_value(6), "the number of nearest neighbor used for computing the density.")
        ("L", po::value<unsigned int>(&L)->default_value(12), "L value used for density computation.")
        ("Lnn", po::value<unsigned int>(&Lnn)->default_value(4), "the starting Lnn value used for dependent point computation.")
        ("Lbuild", po::value<unsigned int>(&Lbuild)->default_value(12), "Retain closest Lbuild number of nodes during the greedy search of construction.")
        ("max_degree", po::value<unsigned int>(&max_degree)->default_value(16), "max_degree value used for constructing the graph.")
        ("alpha", po::value<float>(&alpha)->default_value(1.2), "alpha value")
        ("query_file", po::value<std::string>(&query_file)->required(), "Data set file")
        ("output_file", po::value<std::string>(&output_file)->default_value(""), "Output cluster file")
        ("decision_graph_path", po::value<std::string>(&decision_graph_path)->default_value(""), "Output decision_graph_path")
        ("density_cutoff", po::value<float>(&density_cutoff)->default_value(0), "Density below which points are treated as noise")
        ("dist_cutoff", po::value<float>(&dist_cutoff)->default_value(std::numeric_limits<float>::max()), "Distance below which points are sorted into the same cluster")
        ("bruteforce", po::value<bool>(&bruteforce)->default_value(false), "Whether bruteforce method is used.")
				("method", po::value<Method>(&method)->default_value(Method::Doubling), "Method (Doubling or BlindProbe). Only works when bruteforce=false.")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    } catch(const po::error &e) {
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
	std::cout << "dist_cutoff=" << dist_cutoff << "\n";
	std::cout << "num_thread: " << parlay::num_workers() << std::endl;

	if(bruteforce){
		std::cout << "method= brute force\n";
		DPC::dpc_bruteforce(K, query_file, density_cutoff, dist_cutoff, 
	    output_file, decision_graph_path);
	} else {

    std::cout << "method= " << method << std::endl;
    std::cout << "K=" << K << "\n";
    std::cout << "L=" << L << "\n";
    std::cout << "Lnn=" << Lnn << "\n";
    std::cout << "Lbuild=" << Lbuild << "\n";
    std::cout << "max_degree=" << max_degree << "\n";
    std::cout << "alpha=" << alpha << "\n";

		DPC::dpc(K, L, Lnn, query_file, density_cutoff, dist_cutoff, 
				output_file, decision_graph_path, Lbuild, max_degree, alpha, method);
	}

}
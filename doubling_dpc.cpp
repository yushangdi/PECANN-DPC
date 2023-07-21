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

#include "ParlayANN/algorithms/vamana/neighbors.h"
#include "ParlayANN/algorithms/utils/parse_files.h"
// #include "ParlayANN/algorithms/vamana/index.h"
// #include "ParlayANN/algorithms/utils/types.h"
// #include "ParlayANN/algorithms/utils/beamSearch.h"
// #include "ParlayANN/algorithms/utils/stats.h"


#include "IO.h"

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/slice.h"
#include "parlay/internal/get_time.h"

#include "union_find.h"


// g++ -std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG -IParlayANN/parlaylib/include doubling_dpc.cpp -I/home/ubuntu/boost_1_82_0 -o doubling_dpc -lboost_program_options
namespace po = boost::program_options;

bool report_stats = true;

enum class Method {
	Doubling, BlindProbe
};

// v, i, densities, data_aligned_dim, Lnn, index
template<class T>
std::pair<uint32_t, double> compute_dep_ptr(parlay::sequence<Tvec_point<T>*> data, std::size_t query_id, const std::vector<T>& densities, 
													const size_t data_aligned_dim, const unsigned L, Distance* D){
	// if(L*4 > densities.size()) return densities.size(); // why?
	
	parlay::sequence<Tvec_point<T>*> start_points;
	start_points.push_back(data[query_id]);
	auto [pairElts, dist_cmps] = beam_search<T>(data[query_id], data, 
																					start_points, L, data_aligned_dim, D);
	auto [beamElts, visitedElts] = pairElts;

	double query_density = densities[query_id];
	T* query_ptr = data[query_id]->coordinates.begin();
	float minimum_dist = std::numeric_limits<float>::max();
	uint32_t dep_ptr = densities.size();
	for(unsigned i=0; i<L; i++){
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
	if(dep_ptr == densities.size()){
		return compute_dep_ptr(data, query_id, densities, data_aligned_dim, L*2, D);
	}
	return {dep_ptr, minimum_dist};
}


// v, i, densities, data_aligned_dim, Lnn, index
template<class T>
std::pair<uint32_t, double> compute_dep_ptr_blind_probe(parlay::sequence<Tvec_point<T>*> data, std::size_t query_id, const std::vector<T>& densities, 
													const size_t data_aligned_dim, const unsigned L, Distance* D){
	// if(L*4 > densities.size()) return densities.size(); // why?
	
	parlay::sequence<Tvec_point<T>*> start_points;
	start_points.push_back(data[query_id]);
	auto [pairElts, dist_cmps] = beam_search_blind_probe<T, T>(data[query_id], data, densities,
																					start_points, L, data_aligned_dim, D);
	auto [beamElts, visitedElts] = pairElts;

	double query_density = densities[query_id];
	T* query_ptr = data[query_id]->coordinates.begin();
	float minimum_dist = std::numeric_limits<float>::max();
	uint32_t dep_ptr = densities.size();
	for(unsigned i=0; i<L; i++){
		const auto [id, dist] = beamElts[i];
		if (id == query_id) continue;
		// if(id == densities.size()) break;
		if(densities[id] > query_density || (densities[id] == query_density && id > query_id)){
			if(dist < minimum_dist){
				minimum_dist = dist;
				dep_ptr = id;
			}
		} else {
			std::cout << "Internal error: blind probe retuned invalid points \n.";
		}
	}
	if(dep_ptr == densities.size()){
		return compute_dep_ptr_blind_probe(data, query_id, densities, data_aligned_dim, L*2, D);
	}
	return {dep_ptr, minimum_dist};
}

void dpc(const unsigned K, const unsigned L, const unsigned Lnn, const unsigned num_threads, const std::string& data_path, 
         const std::string& output_path, const std::string& decision_graph_path, const unsigned Lbuild, const unsigned max_degree, const float alpha, Method method ){
	using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::microseconds;
  using T = float;
  parlay::internal::timer t("DPC",report_stats);

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

	auto density_query_pts = parlay::tabulate(data_num, [&] (size_t i) -> Tvec_point<T>* {
      return &points[i];});

  t.next("Load data.");

  using findex = knn_index<T>;
  findex I(max_degree, Lbuild, alpha, data_aligned_dim, D);
  // I.find_approx_medoid(v);
  parlay::sequence<int> inserts = parlay::tabulate(v.size(), [&] (size_t i){
          return static_cast<int>(i);});
  I.build_index(v, inserts);
  t.next("Built index");

	std::cout << v[8]->out_nbh[0] << std::endl;
	if(v[8]->out_nbh[0] == -1){
		std::cout << "node 9 does not have any neighbor\n";
		exit(1);
	}

  if(report_stats){
    auto [avg_deg, max_deg] = graph_stats(v);
    std::cout << "Index built with average degree " << avg_deg << " and max degree " << max_deg << std::endl;
    t.next("stats");
  }

	std::vector<T> densities(data_num);
	auto beamSizeQ = L;
	parlay::parallel_for(0, data_num, [&](size_t i) {
    // parlay::sequence<int> neighbors = parlay::sequence<int>(k);
		// what is cut and limit?
		parlay::sequence<Tvec_point<T>*> start_points;
    start_points.push_back(v[i]);
    auto [pairElts, dist_cmps] = beam_search(density_query_pts[i], v, 
																						start_points, beamSizeQ, data_aligned_dim, D, K);
    auto [beamElts, visitedElts] = pairElts;
		auto less = [&](pid a, pid b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first); };
		auto sorted_nn = parlay::sort(beamElts, less);
		T distance = sorted_nn[0].second;
		if(distance <= 0){
			densities[i] =  std::numeric_limits<T>::max();
		}else{
			densities[i] =  1/distance;
		}
  });
  t.next("Compute density");


  std::vector<std::pair<uint32_t, double>> dep_ptrs(data_num);
	if (method == Method::Doubling){
		parlay::parallel_for(0, data_num, [&](size_t i) {
			dep_ptrs[i] = compute_dep_ptr(v, i, densities, data_aligned_dim, Lnn, D);
		});
	} else if (method == Method::BlindProbe){
		parlay::parallel_for(0, data_num, [&](size_t i) {
			dep_ptrs[i] = compute_dep_ptr_blind_probe(v, i, densities, data_aligned_dim, Lnn, D);
		});
	} else {
		std::cout << "Error: method not implemented " << std::endl;
		exit(1);
	}


  aligned_free(data);
	t.next("Compute dependent points");

  union_find<int> UF(densities.size());
	parlay::parallel_for(0, densities.size(), [&](int i){
		if(dep_ptrs[i].first != densities.size())
			UF.link(i, dep_ptrs[i].first);
	});
	std::vector<int> cluster(densities.size());
	parlay::parallel_for(0, densities.size(), [&](int i){
		cluster[i] = UF.find(i);
	});

	t.next("Find clusters");

    if(output_path != ""){
    	std::ofstream fout(output_path);
    	for (size_t i = 0; i < cluster.size(); i++){
    		fout << cluster[i] << std::endl;
    	}
    	fout.close();
	}

	if(decision_graph_path != ""){    	
    	std::ofstream fout(decision_graph_path);
    	for (size_t i = 0; i < data_num; i++){
    		fout << densities[i] << " " << dep_ptrs[i].second << '\n';
    	}
    }
}




int main(int argc, char** argv){
	std::string query_file, output_file, decision_graph_path;

	// TODO: change to use parse_command_line.h
	po::options_description desc{"Arguments"};
 	try {
	    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file");
	    desc.add_options()("output_file",
                       po::value<std::string>(&output_file)->default_value(""),
                       "Output cluster file");
			desc.add_options()("decision_graph_path",
                       po::value<std::string>(&decision_graph_path)->default_value(""),
                       "Output decision_graph_path");
	    po::variables_map vm;
	    po::store(po::parse_command_line(argc, argv, desc), vm);
    	if (vm.count("help")) {
	    	std::cout << desc;
    		return 0;
	    }
    	po::notify(vm);	
	}catch(const std::exception& ex) {
    	std::cerr << ex.what() << '\n';
	    return -1;
	}
  const unsigned K = 6;
  const unsigned L = 12;
  const unsigned Lnn = 2;
  const unsigned num_threads = 8;
  const unsigned Lbuild = 12; 
  const unsigned max_degree = 16;
  const float alpha = 1.2;

	Method method = Method::Doubling;

	dpc(K, L, Lnn, num_threads, query_file, output_file, decision_graph_path, Lbuild, max_degree, alpha, method);
}
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

namespace {
	void report(double time, std::string str) {
    std::ios::fmtflags cout_settings = std::cout.flags();
    std::cout.precision(4);
    std::cout << std::fixed;
    // std::cout << name << ": ";
    if (str.length() > 0)
      std::cout << str << ": ";
    std::cout << time << std::endl;
    std::cout.flags(cout_settings);
  }
}

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
		auto less = [&](pid a, pid b) {
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

template<class T>
void output(const std::vector<T>& densities, const std::vector<int>& cluster, const std::vector<std::pair<uint32_t, double>>& dep_ptrs, 
						const std::string& output_path, const std::string& decision_graph_path){
    if(output_path != ""){
    	std::ofstream fout(output_path);
    	for (size_t i = 0; i < cluster.size(); i++){
    		fout << cluster[i] << std::endl;
    	}
    	fout.close();
	}

	if(decision_graph_path != ""){    	
		std::cout << "writing decision graph\n";
    	std::ofstream fout(decision_graph_path);
    	for (size_t i = 0; i < densities.size(); i++){
    		fout << densities[i] << " " << dep_ptrs[i].second << '\n';
    	}
    }
}

template<class T>
std::vector<int> cluster_points(std::vector<T>& densities, std::vector<std::pair<uint32_t, double>>& dep_ptrs, 
																float density_cutoff, float dist_cutoff){
  union_find<int> UF(densities.size());
	parlay::parallel_for(0, densities.size(), [&](int i){
		if(dep_ptrs[i].first != densities.size()){ // the max density point
			if(densities[i] > density_cutoff && dep_ptrs[i].second <= dist_cutoff){
				UF.link(i, dep_ptrs[i].first);
			}
		}
	});
	std::vector<int> cluster(densities.size());
	parlay::parallel_for(0, densities.size(), [&](int i){
		cluster[i] = UF.find(i);
	});
	return cluster;
}

void dpc_bruteforce(const unsigned K, const std::string& data_path, float density_cutoff, float distance_cutoff,
         const std::string& output_path, const std::string& decision_graph_path){
  using T = float;
	T* data = nullptr;
	size_t data_num, data_dim, data_aligned_dim;
	load_text_file(data_path, data, data_num, data_dim,
                               data_aligned_dim);
	std::cout<<"data_num: "<<data_num<<std::endl;

  auto points = parlay::sequence<Tvec_point<T>>(data_num);
  parlay::parallel_for(0, data_num, [&] (size_t i) {
    T* start = data + (i * data_aligned_dim);
    T* end = data + ((i+1) * data_aligned_dim);
    points[i].id = i; 
    points[i].coordinates = parlay::make_slice(start, end);  
  });
  Distance* D = new Euclidian_Distance();

  parlay::internal::timer t("DPC");

	std::vector<float> densities(data_num);
	parlay::parallel_for(0, data_num, [&] (size_t i) {
		std::vector<float> dists(data_num);
		for(size_t j=0; j<data_num; j++) dists[j] = D->distance(points[i].coordinates.begin(), points[j].coordinates.begin(), data_dim);
		std::nth_element(dists.begin(), dists.begin()+K-1, dists.end());
		densities[i] = 1/dists[K-1];
	}, 1);

	double density_time = t.next_time();
  report(density_time, "Compute density");

	std::vector<std::pair<uint32_t, double>> dep_ptrs(data_num);
		for(size_t i=0; i<data_num; i++) {
		std::vector<float> dists(data_num, std::numeric_limits<float>::max());
		// TODO: skip noise points
		for(size_t j=0; j<data_num; j++){ 
			if(densities[j] > densities[i] || (densities[j] == densities[i] && j < i))
				dists[j] = D->distance(points[i].coordinates.begin(), points[j].coordinates.begin(), data_dim);
		}
		float m_dist = std::numeric_limits<float>::max();
		size_t id = data_num;
		for(size_t j=0; j<data_num; j++){
			if(dists[j] <= m_dist){
				m_dist = dists[j];
				id = j;
			}
		}
		dep_ptrs[i] = {id, m_dist};
	}
  aligned_free(data);
	double dependent_time = t.next_time();
	report(dependent_time, "Compute dependent points");


	const auto& cluster = cluster_points(densities, dep_ptrs, density_cutoff, distance_cutoff);
	double cluster_time = t.next_time();
	report(cluster_time, "Find clusters");
	report(density_time + dependent_time + cluster_time, "Total");

	output(densities, cluster, dep_ptrs, output_path, decision_graph_path);
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

	// auto density_query_pts = parlay::tabulate(data_num, [&] (size_t i) -> Tvec_point<T>* {
  //     return &points[i];});

  t.next("Load data.");

  using findex = knn_index<T>;
  findex I(max_degree, Lbuild, alpha, data_dim, D);
  // I.find_approx_medoid(v);
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

	Tvec_point<T>** max_density_point = parlay::max_element(v, [&densities](Tvec_point<T>* a, Tvec_point<T>* b){
		if(densities[a->id] == densities[b->id]){
			return a->id < b->id;
		}
		return densities[a->id] < densities[b->id];
	});
	auto max_point_id = max_density_point[0]->id;

  std::vector<std::pair<uint32_t, double>> dep_ptrs(data_num);
	dep_ptrs[max_point_id] = {data_num, -1};
	if (method == Method::Doubling){
		parlay::parallel_for(0, data_num, [&](size_t i) {
			if (i != max_point_id && densities[i] > density_cutoff){ // skip noise points
			dep_ptrs[i] = compute_dep_ptr(v, i, densities, data_aligned_dim, Lnn, D);
			}
		});
	} else if (method == Method::BlindProbe){
		parlay::parallel_for(0, data_num, [&](size_t i) {
			if (i != max_point_id && densities[i] > density_cutoff){ // skip noise points
			dep_ptrs[i] = compute_dep_ptr_blind_probe(v, i, densities, data_aligned_dim, Lnn, D);
			}
		});
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
}




int main(int argc, char** argv){
	std::string query_file, output_file, decision_graph_path;
	float density_cutoff, dist_cutoff;
	bool bruteforce = false;
  unsigned int K = 6;
  unsigned int L = 12;
  unsigned int Lnn = 2;
  unsigned int Lbuild = 12; 
  unsigned int max_degree = 16;
  float alpha = 1.2;

	po::options_description desc("Allowed options");
    desc.add_options()
        ("K", po::value<unsigned int>(&K)->default_value(6), "the number of nearest neighbor used for computing the density.")
        ("L", po::value<unsigned int>(&L)->default_value(12), "L value used for density computation.")
        ("Lnn", po::value<unsigned int>(&Lnn)->default_value(2), "Lnn value used for dependent point computation.")
        ("Lbuild", po::value<unsigned int>(&Lbuild)->default_value(12), "Retain closest Lbuild number of nodes during the greedy search of construction.")
        ("max_degree", po::value<unsigned int>(&max_degree)->default_value(16), "max_degree value used for constructing the graph.")
        ("alpha", po::value<float>(&alpha)->default_value(1.2), "alpha value")
        ("query_file", po::value<std::string>(&query_file)->required(), "Data set file")
        ("output_file", po::value<std::string>(&output_file)->default_value(""), "Output cluster file")
        ("decision_graph_path", po::value<std::string>(&decision_graph_path)->default_value(""), "Output decision_graph_path")
        ("density_cutoff", po::value<float>(&density_cutoff)->default_value(0), "Density below which points are treated as noise")
        ("dist_cutoff", po::value<float>(&dist_cutoff)->default_value(std::numeric_limits<float>::max()), "Distance below which points are sorted into the same cluster")
        ("bruteforce", po::value<bool>(&bruteforce)->default_value(false), "Whether bruteforce method is used.")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }


	Method method = Method::Doubling;

	if(bruteforce){
		std::cout << "using brute force\n";
		dpc_bruteforce(K, query_file, density_cutoff, dist_cutoff, 
	    output_file, decision_graph_path);
	} else {
		dpc(K, L, Lnn, query_file, density_cutoff, dist_cutoff, 
				output_file, decision_graph_path, Lbuild, max_degree, alpha, method);
	}

}
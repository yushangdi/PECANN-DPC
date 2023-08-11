#pragma once

#include <parlay/primitives.h>
#include <limits>
#include <fstream>
#include <iostream>
#include <string>


// The following supports both "link" (a directed union) and "find".
// They are safe to run concurrently as long as there is no cycle among
// concurrent links.   This can be achieved, for example by only linking
// a vertex with lower id into one with higher degree.
// See:  "Internally deterministic parallel algorithms can be fast"
// Blelloch, Fineman, Gibbons, and Shun
// for a discussion of link/find.
template <class vertex>
struct union_find {
  parlay::sequence<std::atomic<vertex>> parents;

  bool is_root(vertex u) {
    return parents[u] < 0;
  }

  // initialize n elements all as roots
  union_find(size_t n) :
      parents(parlay::tabulate<std::atomic<vertex>>(n, [] (long) {
        return -1;})) { }

  vertex find(vertex i) {
    if (is_root(i)) return i;
    vertex p = parents[i];
    if (is_root(p)) return p;

    // find root, shortcutting along the way
    do {
      vertex gp = parents[p];
      parents[i] = gp;
      i = p;
      p = gp;
    } while (!is_root(p));
    return p;
  }

  // Version of union that is safe for parallelism
  // when no cycles are created (e.g. only link from lower density to higher density.
  // Does not use ranks.
  void link(vertex u, vertex v) {
    parents[u] = v;
  }
};



using namespace std;

  template <typename IntType>
    struct ParUF {

    IntType *parents;
    pair<IntType, IntType> *hooks; // the edge that merged comp idx with a higher idx comp
    IntType m_n;
	double *values;
	bool store_value;

      // initialize with all roots marked with -1
    ParUF(IntType n, bool t_store_value = false) {
		m_n = n;
		parents = (IntType *)malloc(sizeof(IntType) * n);
        hooks = (pair<IntType, IntType> *) malloc(sizeof(pair<IntType, IntType>) * n);

		parlay::parallel_for(0,n,[&](IntType i){
            parents[i] = std::numeric_limits<IntType>::max();
            hooks[i] = make_pair(std::numeric_limits<IntType>::max(), std::numeric_limits<IntType>::max());
        }); 
		
		store_value = t_store_value;
		if (store_value){
			values = (double *)malloc(sizeof(double) * n);
		}else{
			values = nullptr;
		}
    }

      void del() {free(parents); free(hooks); if(values) free(values);}

      // Not making parent array volatile improves
      // performance and doesn't affect correctness
      inline IntType find(IntType i) {
	IntType j = i;
	if (parents[j] == std::numeric_limits<IntType>::max()) return j;
	do j = parents[j];
	while (parents[j] < std::numeric_limits<IntType>::max());
	//note: path compression can happen in parallel in the same tree, so
	//only link from smaller to larger to avoid cycles
	IntType tmp;
	while((tmp=parents[i])<j){ parents[i]=j; i=tmp;} 
	return j;
      }

    IntType link(IntType u, IntType v) {
		IntType c_from = u;
		IntType c_to = v;
		while(1){
		u = find(u);
		v = find(v);
		if(u == v) break;
		if(u > v) swap(u,v);
		//if successful, store the ID of the edge used in hooks[u]
		if(hooks[u].first == std::numeric_limits<IntType>::max() && __sync_bool_compare_and_swap(&hooks[u].first,std::numeric_limits<IntType>::max(),c_from)){
			parents[u]=v;
			hooks[u].second=c_to;
			break;
		}
		}
		return parents[u];
    }

	IntType link(IntType u, IntType v, double lv) {
		IntType c_from = u;
		IntType c_to = v;
		while(1){
		u = find(u);
		v = find(v);
		if(u == v) break;
		if(u > v) swap(u,v);
		//if successful, store the ID of the edge used in hooks[u]
		if(hooks[u].first == std::numeric_limits<IntType>::max() && __sync_bool_compare_and_swap(&hooks[u].first,std::numeric_limits<IntType>::max(),c_from)){
			parents[u]=v;
			hooks[u].second=c_to;
			if (store_value) values[u] = lv;
			break;
		}
		}
		return parents[u];
    }

	pair<pair<IntType, IntType>, double> get_edge_value(IntType idx) {
		return make_pair(hooks[idx], values[idx]);
    }	
};
  
// The input density should be in position in the density array to deal with tie breaking.
template <typename T, typename V>
std::pair<std::pair<parlay::sequence<pid>, parlay::sequence<pid>>, size_t> beam_search_blind_probe(
    Tvec_point<T>* p, parlay::sequence<Tvec_point<T>*>& v,const std::vector<V>& densities,
    parlay::sequence<Tvec_point<T>*> starting_points, int beamSize, unsigned d, Distance* D, int k=0, float cut=1.14, int limit=-1) {
  // initialize data structures
  if(limit==-1) limit=v.size();
  T query_density = densities[p->id];
  size_t dist_cmps = 0;
  auto vvc = v[0]->coordinates.begin();
  long stride = v[1]->coordinates.begin() - v[0]->coordinates.begin();
  std::vector<pid> visited;
  auto less = [&](pid a, pid b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first); };
  auto make_pid = [&] (int q) {
      return std::pair{q, D->distance(vvc + q*stride, p->coordinates.begin(), d)};
  };
  auto filter_f = [&](const pid& q){
    if(densities[q.first] > query_density  || (densities[q.first] == query_density && q.first > p->id)) return true;
    return false;
  };
  int bits = std::ceil(std::log2(beamSize*beamSize))-2;
  parlay::sequence<int> hash_table(1 << bits, -1);

  auto pre_frontier = parlay::tabulate(starting_points.size(), [&] (size_t i) {
    return make_pid(starting_points[i]->id);
  });

  dist_cmps += starting_points.size();

  auto frontier = parlay::sort(pre_frontier, less);

  std::vector<pid> unvisited_frontier(beamSize);
  // init filteredNodes
  auto filteredNodes = parlay::filter(frontier, filter_f);
  parlay::sequence<pid> new_filtered(beamSize + v[0]->out_nbh.size()); // why this never overflows?
  parlay::sequence<pid> new_frontier(beamSize + v[0]->out_nbh.size()); // why this never overflows?
  unvisited_frontier[0] = frontier[0];
  int remain = 1;
  int num_visited = 0;

  // terminate beam search when the entire frontier has been visited
  while (remain > 0 && num_visited<limit) {
    // the next node to visit is the unvisited frontier node that is closest to p
    pid currentPid = unvisited_frontier[0];
    Tvec_point<T>* current = v[currentPid.first];
    auto nbh = current->out_nbh.cut(0, size_of(current->out_nbh));
    auto candidates = parlay::filter(nbh,  [&] (int a) {
	     int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
	     if (a == p->id || hash_table[loc] == a) return false;
	     hash_table[loc] = a;
	     return true;});
    auto pairCandidates = parlay::map(candidates, [&] (long c) {return make_pid(c);}, 1000); // what is 1000?
    dist_cmps += candidates.size();
    auto sortedCandidates = parlay::sort(pairCandidates, less);
    auto f_iter = std::set_union(frontier.begin(), frontier.end(),
				 sortedCandidates.begin(), sortedCandidates.end(),
				 new_frontier.begin(), less);
    size_t f_size = std::min<size_t>(beamSize, f_iter - new_frontier.begin());
    if (k > 0 && f_size > k) {
      if(D->id() == "mips"){
        f_size = (std::upper_bound(new_frontier.begin(), new_frontier.begin() + f_size,
				std::pair{0, -cut * new_frontier[k].second}, less)
		- new_frontier.begin());
      }
      else{f_size = (std::upper_bound(new_frontier.begin(), new_frontier.begin() + f_size,
				std::pair{0, cut * new_frontier[k].second}, less)
		- new_frontier.begin());}
    }
    frontier = parlay::tabulate(f_size, [&] (long i) {return new_frontier[i];});


    // insert new candidates into filteredNodes and maintain the best `beamSize` ones.
    auto filteredCandidates = parlay::filter(sortedCandidates, filter_f);
    auto sortedFilteredCandidates = parlay::sort(filteredCandidates, less); // TODO: does filter preserve the order?
    auto ff_iter = std::set_union(filteredNodes.begin(), filteredNodes.end(),
            sortedFilteredCandidates.begin(), sortedFilteredCandidates.end(),
            new_filtered.begin(), less);
    size_t ff_size = std::min<size_t>(beamSize, ff_iter - new_filtered.begin());
    filteredNodes = parlay::tabulate(ff_size, [&] (long i) {return new_filtered[i];});

    visited.insert(std::upper_bound(visited.begin(), visited.end(), currentPid, less), currentPid);
    auto uf_iter = std::set_difference(frontier.begin(), frontier.end(),
				 visited.begin(), visited.end(),
				 unvisited_frontier.begin(), less);
    remain = uf_iter - unvisited_frontier.begin();
    num_visited++;
  }
  return std::make_pair(std::make_pair(filteredNodes, parlay::to_sequence(visited)), dist_cmps);
}

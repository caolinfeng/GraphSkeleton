#ifndef __GRAPH0429_H
#define __GRAPH0429_H


#include <bits/stdc++.h>
#include <boost/random.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

typedef unsigned long long uLL;
typedef long long LL;

class Graph{
private:
    vector< vector<int> > E_;
    vector<int> target_id_;
    vector<bool> is_target_;
    long long num_edge_;
    int num_node_, num_target_, num_component_;

    int d1_, d2_;
    bool *mask_, *mask_bridge_, *mask_corr_;
    uLL *hash_value_, *token_pool_;

private:

    int pos(int i, int j, int p);
    void local_node_filter(int &u, int &d1, int &d2, int *tmp_dis);
    void find_target(int source, int dk, unordered_set<int> &vis);
    void create_find_target_thread(int L, int R, int dk);


public:

    // Constructor
    Graph(np::ndarray edge_index_np, np::ndarray star_np);

    p::dict stat0(int d1, int d2, int dk);

    p::dict stat1(int d1, int d2, int dk);

    np::ndarray extract_skeleton(int d1, int d2, int dk, bool early_stop, int num_worker);

    np::ndarray reconstruct_edge(np::ndarray n_id_np);

    p::tuple reconstruct_reweighted_edge(np::ndarray n_id_np);

    np::ndarray cut_edge_on_skeleton(np::ndarray edge_index_np);

    void get_bridge_and_corr_mask(int d1, int d2);

    np::ndarray get_corr_mask(int d1, int d2);

    np::ndarray get_bridge_mask(int d1, int d2);

    void drop_corr();

    np::ndarray nearest_target();
};

const char* init();

#endif

#include "graph.h"

p::dict Graph::stat0(int d1, int d2, int dk) {

    unordered_set<int> vis;
    p::dict ret;

    for (int i=0;i<num_node_;i++) {
        if (!is_target_[i] && mask_[i]) {
            vis.clear();
            vis.insert(i);
            string cur = "";
            queue<int> qu, qd;
            qu.push(i);
            qd.push(0);
            while (!qu.empty()) {
                int u = qu.front(); qu.pop();
                int d = qd.front(); qd.pop();
                if (is_target_[u]) {
                    cur += to_string(d);
                    continue;
                }
                if (d >= dk) continue;
                for (int i=0;i<(int)E_[u].size();i++) {
                    int v = E_[u][i];
                    if (vis.find(v)==vis.end()) {
                        vis.insert(v);
                        qu.push(v);
                        qd.push(d+1);
                    }
                }
            }
            ret[cur] = ret.get(cur, 0) + 1;
        }
    }
    return ret;
}


p::dict Graph::stat1(int d1, int d2, int dk) {

    unordered_set<int> vis;
    p::dict ret;

    for (int i=0;i<num_node_;i++) {
        if (!is_target_[i] && mask_[i]) {
            vis.clear();
            vis.insert(i);
            string cur = "";
            queue<int> qu, qd;
            qu.push(i);
            qd.push(0);
            while (!qu.empty()) {
                int u = qu.front(); qu.pop();
                int d = qd.front(); qd.pop();
                if (is_target_[u]) {
                    cur += to_string(d);
                    continue;
                }
                if (d >= dk) continue;
                for (int i=0;i<(int)E_[u].size();i++) {
                    int v = E_[u][i];
                    if (vis.find(v)==vis.end()) {
                        vis.insert(v);
                        qu.push(v);
                        qd.push(d+1);
                    }
                }
            }
            ret[cur] = ret.get(cur, 0) + 1;
        }
    }
    return ret;
}


/*
p::dict Graph::stat1(int d1, int d2, int dk) {
    // Check "n_id_np"
    int nd_n_id_np = n_id_np.get_nd();
    if (nd_edge_index_np != 1)
        throw std::runtime_error("\"n_id\" must be 1-dimensional numpy.ndarray. ");
    if (edge_index_np.get_dtype() != np::dtype::get_builtin<int>())
        throw std::runtime_error("\"edge_index\" must be int32 numpy.ndarray. ");

    // Check "edge_index"
    int nd_edge_index_np = edge_index_np.get_nd();
    if (nd_edge_index_np != 2 || edge_index_np.shape(0) != 2)
        throw std::runtime_error("\"edge_index\" must be 2-dimensional numpy.ndarray shapes like [2, num_edge]. ");
    if (edge_index_np.get_dtype() != np::dtype::get_builtin<int>())
        throw std::runtime_error("\"edge_index\" must be int32 numpy.ndarray. ");

    // check "star"
    if (star_np.get_dtype() != np::dtype::get_builtin<bool>())
        throw std::runtime_error("\"target_node\" must be bool numpy.ndarray. ");


    unordered_set<int> vis;
    p::dict ret;

    for (int i=0;i<num_node;i++) {
        if (!is_target[i]) {
            vis.clear();
            vis.insert(i);
            string cur = "";
            queue<int> qu, qd;
            qu.push(i);
            qd.push(0);
            while (!qu.empty()) {
                int u = qu.front(); qu.pop();
                int d = qd.front(); qd.pop();
                if (is_target[u]) {
                    cur += to_string(d);
                    continue;
                }
                if (d >= dk) continue;
                for (int i=0;i<(int)E[u].size();i++) {
                    int v = E[u][i];
                    if (vis.find(v)==vis.end()) {
                        vis.insert(v);
                        qu.push(v);
                        qd.push(d+1);
                    }
                }
            }
            ret[i] = cur;
        }
    }
    return ret;
}

*/

#include "graph.h"

int Graph::pos(int i, int j, int p) {
    return i*p+j;
}

void Graph::find_target(int source, int dk, unordered_set<int> &vis) {
    queue<int> qu, qd;
    qu.push(source);
    qd.push(0);
    vis.clear();
    vis.insert(source);
    while (!qu.empty()) {
        int u = qu.front(); qu.pop();
        int d = qd.front(); qd.pop();
        if (is_target_[u]) {
            int tid = target_id_[u];
            hash_value_[source] ^= token_pool_[ pos(tid, 0, dk) ];
            continue;
        }
        if (d >= dk) continue;
        for (int i=0;i<(int)E_[u].size();i++) {
            int v = E_[u][i];
            if (mask_[v] && vis.find(v)==vis.end()) {
                vis.insert(v);
                qu.push(v);
                qd.push(d+1);
            }
        }
    }
    return ;
}

void Graph::create_find_target_thread(int L, int R, int dk) {
    unordered_set<int> vis;
    for (int i=L;i<R;i++) {
        if (!is_target_[i] && mask_[i] == true) {
            find_target(i, dk, vis);
        }
    }
    return ;
}


np::ndarray Graph::extract_skeleton(int d1, int d2, int dk, bool early_stop = true, int num_worker = 1) {
    printf("::: Params: d1(%d) d2(%d) dk(%d)\n", d1, d2, dk);
    token_pool_ = new uLL[num_target_ * dk];
    hash_value_ = new uLL[num_node_];

    int *d = new int[num_node_];
    int *n_id = new int[num_node_];
    num_component_ = 0;

    for (int i=0;i<num_node_;i++) {
        n_id[i] = 0;
        hash_value_[i] = 0;
    }

    boost::random::mt19937_64 rand_gen(time(0));
    for (int i=0;i<num_target_;i++)
        for (int j=0;j<dk;j++)
            token_pool_[pos(i, j, dk)] = rand_gen();

    if (mask_ == NULL || d1_!=d1 || d2_!=d2){
        get_bridge_and_corr_mask(d1, d2);
    }

    int tmp_cnt_nodes = 0;
    for (int i=0;i<num_node_;i++) {
        if (mask_[i]) ++tmp_cnt_nodes;
    }
    cerr << "tmp_cnt_nodes: " << tmp_cnt_nodes << endl;

    // Multi-thread for finding target nodes from each background nodes.
    std::vector<std::thread> thread_pool_;
    int num_block_node = num_node_ / num_worker + 1;
    for (int i=0;i<num_worker;i++) {
        int L = i * num_block_node;
        int R = (i+1) * num_block_node;
        R = min(R, num_node_);
        thread_pool_.push_back( std::thread( &Graph::create_find_target_thread, this, L, R, dk ) );
    }
    for (int i=0;i<(int)thread_pool_.size();i++) thread_pool_[i].join();
    std::cerr << "find_target Done." << std::endl;


    for (int i=0;i<num_node_;i++) d[i] = i;
    sort(d, d+num_node_, [this](uLL p1, uLL p2){
        if (mask_[p1] != mask_[p2])
            return mask_[p1] < mask_[p2];
        return hash_value_[p1] < hash_value_[p2];
    });

    for (int i=0;i<num_node_;i++) if (is_target_[i]) {
        n_id[i] = num_component_;
        num_component_++;
    }

    for (int i=0;i<num_node_;i++) {
        if (!mask_[i]) {
            n_id[i] = -1;
        }
    }

    for (int i=0;i<num_node_;i++) {
        if (!is_target_[ d[i] ] && mask_[ d[i] ]) {
            n_id[ d[i] ] = num_component_;
            if (i == num_node_-1 || hash_value_[d[i]] != hash_value_[d[i+1]]) {
                num_component_++;
            }
        }
    }


    delete d;
    delete token_pool_;
    delete hash_value_;

    np::dtype dt = np::dtype::get_builtin<int>();
    p::tuple shape = p::make_tuple(num_node_);
    p::tuple stride = p::make_tuple(sizeof(int));
    p::object own;

    np::ndarray n_id_np = np::from_data(n_id, dt, shape, stride, own);
    np::ndarray output_array_np = n_id_np.copy();
    delete n_id;
    return output_array_np;
}


np::ndarray Graph::reconstruct_edge(np::ndarray n_id_np) {
    int *n_id = reinterpret_cast<int *>(n_id_np.get_data());
    set<pair<int, int>> edge_set;
    vector<int> edge_ret_u, edge_ret_v;

    for (int i=0;i<num_node_;i++)
        for (int j=0;j<(int)E_[i].size();j++) {
            int u = n_id[i];
            int v = n_id[E_[i][j]];
            if (u == v) continue;
            if (u == -1 || v == -1) continue;
            if (u > v) swap(u, v);
            auto pair_uv = make_pair(u, v);
            if (edge_set.find(pair_uv) == edge_set.end()) {
                edge_ret_u.push_back(u);
                edge_ret_v.push_back(v);
                edge_set.insert(pair_uv);
            }
        }


    int num_n_edge = (int)edge_ret_u.size();
    int *n_edge_index = new int[2*num_n_edge];
    for (LL i=0;i<num_n_edge;i++) {
        n_edge_index[i] = edge_ret_u[i];
        n_edge_index[i+num_n_edge] = edge_ret_v[i];
    }

    np::dtype dt = np::dtype::get_builtin<int>();
    p::tuple shape = p::make_tuple(2, num_n_edge);
    p::tuple stride = p::make_tuple(sizeof(int)*num_n_edge, sizeof(int));
    p::object own;
    np::ndarray n_edge_index_np = np::from_data(n_edge_index, dt, shape, stride, own);
    np::ndarray output_array_np = n_edge_index_np.copy();

    delete n_edge_index;
    return output_array_np;
}


p::tuple Graph::reconstruct_reweighted_edge(np::ndarray n_id_np) {
    int *n_id = reinterpret_cast<int *>(n_id_np.get_data());
    map<pair<int, int>, double> weight;
    set<int> vis;
    for (int s=0;s<num_node_;s++) {
        if (is_target_[s]) {
            vis.clear();
            queue<int> q, d;
            queue<double> f;
            q.push(s);
            f.push(1.0);
            d.push(0);
            vis.insert(s);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                int dis = d.front(); d.pop();
                double w = f.front(); f.pop();
                if (dis > 2) break;
                for (int i=0;i<(int)E_[u].size();i++) {
                    int v = E_[u][i];
                    if (n_id[v] == -1) continue;
                    if (vis.find(v) != vis.end()) continue;
                    if (dis>=1 && n_id[v] != n_id[u]) continue;
                    q.push(v);
                    f.push(w / 2.0);
                    vis.insert(v);
                    int nu = n_id[s], nv = n_id[v];
                    if (nu == -1 || nv == -1) continue;
                    if (nu > nv) swap(nu, nv);
                    auto tmp = make_pair(nu, nv);
                    if (weight.find(tmp) == weight.end()) {
                        weight[tmp] = w;
                    } else {
                        weight[tmp] += w;
                    }
                }
            }
        }
    }

    // Background-Background
    for (int i=0;i<num_node_;i++)
        for (int j=0;j<(int)E_[i].size();j++) {
            int u = i;
            int v = E_[i][j];
            if (!is_target_[u] && !is_target_[v]) continue;
            u = n_id[u], v = n_id[v];
            if (u == v) continue;
            if (u == -1 || v == -1) continue;
            if (u > v) swap(u, v);
            auto pair_uv = make_pair(u, v);
            if (weight.find(pair_uv) == weight.end()) {
                weight[pair_uv] = 1.0;
            }
        }

    // Target-Target
    for (int i=0;i<num_node_;i++)
        for (int j=0;j<(int)E_[i].size();j++) {
            int u = i;
            int v = E_[i][j];
            if (is_target_[u] && is_target_[v]) continue;
            u = n_id[u], v = n_id[v];
            if (u == v) continue;
            if (u == -1 || v == -1) continue;
            if (u > v) swap(u, v);
            auto pair_uv = make_pair(u, v);
            if (weight.find(pair_uv) == weight.end()) {
                weight[pair_uv] = 1.0;
            }
        }


    size_t num_n_edge = weight.size();
    int *n_edge_index = new int[2*num_n_edge];
    double *n_edge_weight = new double[num_n_edge];
    size_t cur_num_edge = 0;
    for (auto iter=weight.begin(); iter!=weight.end(); iter++) {
        auto pair_uv = iter->first;
        double w = iter->second;
        n_edge_index[cur_num_edge] = pair_uv.first;
        n_edge_index[cur_num_edge+num_n_edge] = pair_uv.second;
        n_edge_weight[cur_num_edge] = w;
        cur_num_edge++;
    }

    np::dtype dt_1 = np::dtype::get_builtin<int>();
    p::tuple shape_1 = p::make_tuple(2, num_n_edge);
    p::tuple stride_1 = p::make_tuple(sizeof(int)*num_n_edge, sizeof(int));
    p::object own_1;
    np::ndarray n_edge_index_np = np::from_data(n_edge_index, dt_1, shape_1, stride_1, own_1);
    np::ndarray output_1_np = n_edge_index_np.copy();
    delete n_edge_index;

    np::dtype dt_2 = np::dtype::get_builtin<double>();
    p::tuple shape_2 = p::make_tuple(num_n_edge);
    p::tuple stride_2 = p::make_tuple(sizeof(double));
    p::object own_2;
    np::ndarray n_edge_weight_np = np::from_data(n_edge_weight, dt_2, shape_2, stride_2, own_2);
    np::ndarray output_2_np = n_edge_weight_np.copy();
    delete n_edge_weight;

    auto ret = p::make_tuple(output_1_np, output_2_np);
    return ret;
}


np::ndarray Graph::cut_edge_on_skeleton(np::ndarray edge_index_np) {
    vector<bool> is_bridge_node(num_node_);
    int *edge_index = reinterpret_cast<int *>(edge_index_np.get_data());

    for (int u=0;u<num_node_;u++) {
        if (!is_target_[u]) {
            int cnt_target_neighbors = 0;
            for (int i=0;i<(int)E_[u].size();i++) {
                int v = E_[u][i];
                if (is_target_[v]) {
                    cnt_target_neighbors++;
                }
            }
            if (cnt_target_neighbors >= 2) {
                is_bridge_node[u] = 1;
            } else {
                is_bridge_node[u] = 0;
            }
        }
    }

    for (LL i=0;i<num_edge_;i++) {
        int u = edge_index[i];
        int v = edge_index[num_edge_+i];
        mask_[i] = true;
        if (is_target_[u] && is_bridge_node[v]) {
            mask_[i] = false;
        }
        if (is_bridge_node[u] && is_target_[v]) {
            mask_[i] = false;
        }
    }

    np::dtype dt = np::dtype::get_builtin<bool>();
    p::tuple shape = p::make_tuple(num_edge_);
    p::tuple stride = p::make_tuple(sizeof(bool));
    p::object own;
    np::ndarray mask_np = np::from_data(mask_, dt, shape, stride, own);
    np::ndarray output_array_np = mask_np.copy();
    return output_array_np;
}


void Graph::local_node_filter(int &u, int &d1, int &d2, int *tmp_dis) {
    queue<int> qu, qd;
    qu.push(u);
    qd.push(0);
    while (!qu.empty()) {
        int u = qu.front(); qu.pop();
        int d = qd.front(); qd.pop();
        if (d <= d1) mask_[u] = true;
        if (d >= d2) continue;
        for (int i=0;i<(int)E_[u].size();i++) {
            int v = E_[u][i];
            if (tmp_dis[v]+d+1 <= d2) {
                mask_bridge_[v] = true;
            }
            if (tmp_dis[v] > d+1) {
                tmp_dis[v] = d+1;
            }
            qu.push(v);
            qd.push(d+1);
        }
    }
    return ;
}


/*************************************************
Function:       get_bridge_and_corr_mask
*************************************************/
void Graph::get_bridge_and_corr_mask(int d1, int d2) {
    auto start_time = std::chrono::system_clock::now();
    cerr << "Get the correlation node and bridge node... ";


    int *vis = new int[num_node_];    // number of visit
    int *prev = new int[num_node_];
    int *dist = new int[num_node_];  // shortest path
    memset(vis, 0, sizeof(int)*num_node_);
    memset(prev, 0, sizeof(int)*num_node_);
    for (int i=0;i<num_node_;i++) dist[i] = d2+1; // d2+1 is INF for local search.

    // malloc space for mask_
    if (mask_ != NULL) delete mask_;
    mask_ = new bool[num_node_];
    memset(mask_, 0, sizeof(bool)*num_node_);

    // malloc space for mask_bridge_
    if (mask_bridge_ == NULL) delete mask_bridge_;
    mask_bridge_ = new bool[num_node_];
    memset(mask_bridge_, 0, sizeof(bool)*num_node_);

    // malloc space for mask_corr_
    if (mask_corr_ == NULL) delete mask_corr_;
    mask_corr_ = new bool[num_node_];
    memset(mask_corr_, 0, sizeof(bool)*num_node_);


    queue<int> qu, qs, qd;
    for (int i=0;i<num_node_;i++) {
        if (is_target_[i]) {
            qu.push(i);
            qs.push(i);
            qd.push(0);
        }
    }
    while (!qu.empty()) {
        int u = qu.front(); qu.pop();
        int s = qs.front(); qs.pop();
        int d = qd.front(); qd.pop();
        for (int i=0;i<(int)E_[u].size();i++) {
            int v = E_[u][i];
            if (is_target_[v]) continue;
            if (vis[v] == 0) {
                vis[v] = 1;
                dist[v] = d + 1;
                prev[v] = s;
                qu.push(v);
                qs.push(s);
                qd.push(d+1);
                if (dist[v] <= d1) mask_corr_[v] = 1;
            } else if (vis[v] == 1 && prev[v] != s) {
                vis[v] = 2;
                qu.push(v);
                qs.push(s);
                qd.push(d+1);
                if (dist[v] + d + 1 <= d2) {
                    mask_corr_[v] = 0;
                    mask_bridge_[v] = 1;
                }
            }
        }
    }

    for (int i=0;i<num_node_;i++) {
        if (mask_corr_[i] || mask_bridge_[i] || is_target_[i]) {
            mask_[i] = 1;
        }
    }

    d1_ = d1, d2_ = d2;
    auto end_time = std::chrono::system_clock::now();
    cerr << "Done. [" << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() << "s]" << endl;
}

np::ndarray Graph::get_corr_mask(int d1, int d2) {
    if (mask_ == NULL || mask_corr_ == NULL || d1_!=d1 || d2_!=d2) {
        get_bridge_and_corr_mask(d1, d2);
    }

    np::dtype dt = np::dtype::get_builtin<bool>();
    p::tuple shape = p::make_tuple(num_node_);
    p::tuple stride = p::make_tuple(sizeof(bool));
    p::object own;
    np::ndarray mask_corr_np = np::from_data(mask_corr_, dt, shape, stride, own);
    np::ndarray output_array_np = mask_corr_np.copy();
    return output_array_np;
}

np::ndarray Graph::get_bridge_mask(int d1, int d2) {
    if (mask_ == NULL || mask_bridge_ == NULL || d1_!=d1 || d2_!=d2) {
        get_bridge_and_corr_mask(d1, d2);
    }

    np::dtype dt = np::dtype::get_builtin<bool>();
    p::tuple shape = p::make_tuple(num_node_);
    p::tuple stride = p::make_tuple(sizeof(bool));
    p::object own;
    np::ndarray mask_bridge_np = np::from_data(mask_bridge_, dt, shape, stride, own);
    np::ndarray output_array_np = mask_bridge_np.copy();
    return output_array_np;
}

// 计算每个mask=true的点最近邻居
np::ndarray Graph::nearest_target() {
    if (mask_ == NULL || mask_corr_ == NULL) {
        throw std::runtime_error("Call get_corr_mask before drop correlation nodes.");
    }

    bool *vis = new bool[num_node_];
    int *nt = new int[num_node_];
    memset(vis, 0, sizeof(bool) * num_node_);
    for (int i=0;i<num_node_;i++) nt[i] = -1;

    // Enqueue all target nodes.
    queue<int> qs, qu, qd;
    for (int i=0;i<num_node_;i++) {
        if (is_target_[i]) {
            qs.push(i);
            qu.push(i);
            qd.push(0);
            vis[i] = true;
            nt[i] = i;
        }
    }

    while (!qu.empty()) {
        int s = qs.front(); qs.pop();
        int u = qu.front(); qu.pop();
        int d = qd.front(); qd.pop();
        for (int j=0;j<(int)E_[u].size();j++) {
            int v = E_[u][j];
            if (mask_[v] && !vis[v]) {
                vis[v] = true;
                nt[v] = u;
                qs.push(s);
                qu.push(v);
                qd.push(d+1);
            }
        }
    }

    delete vis;
    np::dtype dt = np::dtype::get_builtin<int>();
    p::tuple shape = p::make_tuple(num_node_);
    p::tuple stride = p::make_tuple(sizeof(int));
    p::object own;
    np::ndarray nt_np = np::from_data(nt, dt, shape, stride, own);
    np::ndarray output_array_np = nt_np.copy();
    delete nt;
    return output_array_np;

}

void Graph::drop_corr() {
    if (mask_ == NULL || mask_corr_ == NULL) {
        throw std::runtime_error("Call get_corr_mask before drop correlation nodes.");
    }
    for (int i=0;i<num_node_;i++) {
        if (mask_[i] && mask_corr_[i]) {
            mask_[i] = false;
        }
    }
    return ;
}

#include "graph.h"


/*************************************************
Function:       init
Description:    Before calling this library, you should call this initialization function to initialize
                the python settings in the boost library, otherwise an error may be caused.
*************************************************/
const char* init() {
    Py_Initialize();
    np::initialize();
    return "Init.";
}


/*************************************************
Function:       Graph
Description:    Constructor for Graph.
Input:          edge_index_np: A 2-d numpy.ndarray shapes like [2, num_edge_] indicates edge index for the original graph.
                star_np: A 1-d bool numpy.ndarray marks the important nodes.
*************************************************/
Graph::Graph(np::ndarray edge_index_np, np::ndarray star_np) {
    // Check "edge_index"
    int nd_edge_index_np = edge_index_np.get_nd();
    if (nd_edge_index_np != 2 || edge_index_np.shape(0) != 2)
        throw std::runtime_error("\"edge_index\" must be 2-dimensional numpy.ndarray shapes like [2, num_edge_]. ");
    if (edge_index_np.get_dtype() != np::dtype::get_builtin<int>())
        throw std::runtime_error("\"edge_index\" must be int32 numpy.ndarray. ");

    // check "star"
    if (star_np.get_dtype() != np::dtype::get_builtin<bool>())
        throw std::runtime_error("\"target_node\" must be bool numpy.ndarray. ");


    num_node_ = 0;
    num_edge_ = edge_index_np.shape(1);
    int *edge_index = reinterpret_cast<int *>(edge_index_np.get_data());
    bool *star = reinterpret_cast<bool *>(star_np.get_data());
    for (LL i=0;i<2*num_edge_;i++) {
        if (edge_index[i] < 0) {
            throw std::runtime_error("Negative node_id in \"edge_index\". ");
        }
        num_node_ = max(num_node_, edge_index[i]+1);
    }

    E_.resize(num_node_);
    is_target_.resize(num_node_);
    target_id_.resize(num_node_);

    num_target_ = 0;

    for (LL i=0;i<num_edge_;i++) {
        int u = edge_index[i];
        int v = edge_index[num_edge_+i];
        E_[u].push_back(v);
        E_[v].push_back(u);
    }

    for (int i=0;i<num_node_;i++) {
        if (star[i]) {
            is_target_[i] = true;
            target_id_[i] = num_target_;
            num_target_++;
        } else {
            is_target_[i] = false;
        }
    }

    // If mask_ [x] =false, then x will be dropped in the finally extracted skeleton graph.
    // All nodes are available at first.
    mask_ = new bool[num_node_];
    for (int i=0;i<num_node_;i++) mask_[i] = true;


    mask_bridge_ = NULL;
    hash_value_ = NULL;
    token_pool_ = NULL;

    d1_ = 0, d2_= 0;

    // Output on screen
    cerr << "::: Graph(num_node: " << num_node_ << ", num_edge_: " << num_edge_ <<
        ", num_target: " << num_target_ << ")." << endl;
}

BOOST_PYTHON_MODULE(graph_skeleton)
{
    boost::python::def("init", init);
    boost::python::class_<Graph>("Graph", boost::python::init<np::ndarray, np::ndarray>())
        .def("extract_skeleton", &Graph::extract_skeleton)
        .def("reconstruct_edge", &Graph::reconstruct_edge)
        .def("reconstruct_reweighted_edge", &Graph::reconstruct_reweighted_edge)
        .def("cut_edge_on_skeleton", &Graph::cut_edge_on_skeleton)
        .def("stat0", &Graph::stat0)
        .def("stat1", &Graph::stat1)
        .def("get_bridge_and_corr_mask", &Graph::get_bridge_and_corr_mask)
        .def("get_corr_mask", &Graph::get_corr_mask)
        .def("get_bridge_mask", &Graph::get_bridge_mask)
        .def("nearest_target", &Graph::nearest_target)
        .def("drop_corr", &Graph::drop_corr)
    ;
}






    /*************************************************
Function:       // 函数名称
Description:    // 函数功能、性能等的描述
Calls:          // 被本函数调用的函数清单
Table Accessed: // 被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated: // 被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:          // 输入参数说明，包括每个参数的作
                  // 用、取值说明及参数间关系。
Output:         // 对输出参数的说明。
Return:         // 函数返回值的说明
Others:         // 其它说明
*************************************************/

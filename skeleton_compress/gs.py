import time
import numpy as np
import graph_skeleton
# from tqdm import tqdm

def reconstruct_x(x, n_id):
    print(f"Reconstruct feature({x.shape})... ", end=" ", flush=True)
    start_time = time.perf_counter()
    num_node, feature_dim = x.shape
    num_n_node = np.max(n_id) + 1
    n_x = np.zeros((num_n_node, feature_dim), dtype=x.dtype)
    cnt_x = np.zeros((num_n_node, ), dtype=np.int32)
    for i in range(num_node):
        if n_id[i] != -1:
            cnt_x[n_id[i]] += 1
            n_x[n_id[i]] += x[i]
    for i in range(num_n_node):
        n_x[i] /= cnt_x[i]
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")
    return n_x, cnt_x

def mapping_label(y, n_id):
    print(f"Mapping label ({y.shape})... ", end=" ", flush=True)
    start_time = time.perf_counter()
    num_node = np.max(n_id) + 1
    n_y = np.full((num_node, ), -1, dtype=y.dtype)
    for i, l in enumerate(y):
        n_y[ n_id[i] ] = l
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")
    return n_y

def mapping_mask(mask, n_id):
    # print(f"Mapping mask ({mask.shape})... ", end=" ", flush=True)
    start_time = time.perf_counter()
    n_mask = []
    for i in mask:
        n_mask.append(n_id[i])
    n_mask = np.array(n_mask, dtype=mask.dtype)
    # print(f"Done! [{time.perf_counter() - start_time:.2f}s]")
    return n_mask


def zip_nodes(edge_index, target_node, d1=1, d2=1, dk=1):
    '''
    A node is selected if and only if there is a path connecting two important nodes whose length<=k and
    the node is on this path.
    :param edge_index: np.array with the shape of (num_of_edge*2)
    :param star: A np.array with index of important node.
    :param k: Threshold length.
    :return: A bool np.array. If the mask[i] is true, it means that the i-th node
             is selected, and the important nodes are selected by default.
    '''
    print(f"compress nodes (num_edge: {edge_index.shape[1]})... ", end=" ", flush=True)
    start_time = time.perf_counter()
    n = int(np.max(edge_index) + 1)        # Num of vertex
    edge_index = edge_index.astype(np.int32)

    tmp = np.full((n,), False, dtype=np.bool_)
    tmp[target_node] = True
    target_node = tmp

    graph_skeleton.greet()
    g = graph_skeleton.Graph(edge_index, target_node)
    n_id = g.zip(d1, d2, dk, 32)
    np.save("n_id.npy", n_id)
    print('n_id.npy')
    n_edge_index = g.reconstruct_edge(n_id)
    # print(f"n_id: {n_id}")
    # print(f"n_edge_index: {n_edge_index}")
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")
    return n_id, n_edge_index

def zip_graph(x, y, edge_index, train_mask, valid_mask, test_mask, d1=2, d2=2, dk=2):

    ret = {}
    star = np.concatenate((train_mask, valid_mask, test_mask), axis=0)
    print(f"Graph_1: #V: {x.shape[0]}, #E: {edge_index.shape[1]}")

    # === simple test ===
    # star = np.array([0, 1])
    # edge_index = np.array([[1, 1, 1, 4, 5, 1, 6, 1, 7, 1, 8],
    #                        [2, 3, 4, 5, 0, 6, 0, 7, 0, 8, 0]])

    n_id, n_edge_index = zip_nodes(edge_index, star, d1, d2, dk)
    n_edge_index = n_edge_index.astype(np.int64)
    n_x, cnt_x = reconstruct_x(x, n_id)
    n_y = mapping_label(y, n_id)
    n_train_mask = mapping_mask(train_mask, n_id)
    n_valid_mask = mapping_mask(valid_mask, n_id)
    n_test_mask = mapping_mask(test_mask, n_id)
    print(f"Graph_2: #V: {n_x.shape[0]}, #E: {n_edge_index.shape[1]}")
    return {'x':n_x,
            'y':n_y,
            'edge_index':n_edge_index,
            'train_mask':n_train_mask,
            'valid_mask':n_valid_mask,
            'test_mask':n_test_mask,
            'cnt_x': cnt_x
           }

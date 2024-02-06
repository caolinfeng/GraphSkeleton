#!/usr/bin/env python
# coding: utf-8

# In[]:

import graph_skeleton
from dataloader import dataloader
from gs import *
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import time

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument('--dataset', default='dgraph',
                    help='dataset for compression')
parser.add_argument('--cut', default='gamma', choices = ['alpha','beta','gamma'],
                    help='skeleton compreesion strategy')
parser.add_argument('--d', default=[2,1], nargs='+', type=int,
                        help='node fetching depth')
args = parser.parse_args()


# compression grpah

# In[]:

data, root = dataloader(args.dataset)

x = data['x']
y = data['y']
edge_index = data['edge_index'].T

train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask = data['test_mask']
star = np.concatenate((train_mask, valid_mask, test_mask), axis=0)


# In[]:

if args.cut == 'alpha':
    print('*'*10, 'skeleton-alpha', '*'*10)

    print(f"Build... ", end=" ", flush=True)
    start_time = time.perf_counter()
    graph_skeleton.init()
    edge_index32 = edge_index.astype(np.int32)
    star32 = star.astype(np.int32)
    num_node = np.max(edge_index) + 1
    tmp = np.full((num_node,), False, dtype=np.bool_)
    tmp[star] = True
    star32 = tmp
    print(f"edge_index: {edge_index32.shape}")
    print(f"star: {star32.shape}")
    g = graph_skeleton.Graph(edge_index32, star32)
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

    print(f"Compressing... ", end=" ", flush=True)
    start_time = time.perf_counter()
    n_id = g.extract_skeleton(args.d[1], args.d[0], 2, True, 16)
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

    print(f"Reconstruct edge... ", end=" ", flush=True)
    start_time = time.perf_counter()
    # n_edge_index = g.reconstruct_edge(n_id)
    n_edge_index, n_edge_weight = g.reconstruct_reweighted_edge(n_id)
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

    n_x, cnt_x = reconstruct_x(x, n_id)
    n_y = mapping_label(y, n_id)

    n_train_mask = mapping_mask(train_mask, n_id)
    n_valid_mask = mapping_mask(valid_mask, n_id)
    n_test_mask = mapping_mask(test_mask, n_id)
    n_star = mapping_mask(star, n_id)

    print('-'*20)
    print(f'| Skeleton-alpha | #V: {n_x.shape[0]} | #E: {n_edge_index.shape[1]} | #Target: {n_star.shape[0]} |')
    print('BCR: {:.3f}'.format(1-(x.shape[0]-n_x.shape[0])/(x.shape[0]-star.shape[0])))

    skeleton_data = { 'x': n_x, 'y': n_y, 
                'edge_index': n_edge_index, 'edge_weight': n_edge_weight, 
                'train_mask': n_train_mask, 'valid_mask': n_valid_mask, 'test_mask': n_test_mask}
    np.save(root + '/skeleton_alpha.npy', skeleton_data)



# In[]:

elif args.cut == 'beta':

    print('*'*10, 'skeleton-beta', '*'*10)

    print(f"Build... ", end=" ", flush=True)
    start_time = time.perf_counter()
    graph_skeleton.init()
    edge_index32 = edge_index.astype(np.int32)
    star32 = star.astype(np.int32)
    num_node = np.max(edge_index) + 1
    tmp = np.full((num_node,), False, dtype=np.bool_)
    tmp[star] = True
    star32 = tmp
    print(f"edge_index: {edge_index32.shape}")
    print(f"star: {star32.shape}")
    g = graph_skeleton.Graph(edge_index32, star32)
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

    print(f"Compressing... ", end=" ", flush=True)
    start_time = time.perf_counter()
    n_id = g.extract_skeleton(args.d[1], args.d[0], 15, True, 16)
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

    print(f"Reconstruct edge... ", end=" ", flush=True)
    start_time = time.perf_counter()
    # n_edge_index = g.reconstruct_edge(n_id)
    n_edge_index, n_edge_weight = g.reconstruct_reweighted_edge(n_id)
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

    n_x, cnt_x = reconstruct_x(x, n_id)
    n_y = mapping_label(y, n_id)

    n_train_mask = mapping_mask(train_mask, n_id)
    n_valid_mask = mapping_mask(valid_mask, n_id)
    n_test_mask = mapping_mask(test_mask, n_id)
    n_star = mapping_mask(star, n_id)

    print('-'*20)
    print(f'| Skeleton-beta | #V: {n_x.shape[0]} | #E: {n_edge_index.shape[1]} | #Target: {n_star.shape[0]} |')
    print('BCR: {:.3f}'.format(1-(x.shape[0]-n_x.shape[0])/(x.shape[0]-star.shape[0])))

    skeleton_data = { 'x': n_x, 'y': n_y, 
                'edge_index': n_edge_index, 'edge_weight': n_edge_weight, 
                'train_mask': n_train_mask, 'valid_mask': n_valid_mask, 'test_mask': n_test_mask}
    np.save(root + '/skeleton_beta.npy', skeleton_data)




# In[ ]:

elif args.cut == 'gamma':

    # allfliation merge
    print('*'*10, 'skeleton-gamma', '*'*10)
    print('first use beta strategy')
    start_time = time.perf_counter()
    graph_skeleton.init()
    edge_index32 = edge_index.astype(np.int32)
    star32 = star.astype(np.int32)
    num_node = np.max(edge_index) + 1
    tmp = np.full((num_node,), False, dtype=np.bool_)
    tmp[star] = True
    star32 = tmp
    g = graph_skeleton.Graph(edge_index32, star32)

    start_time = time.perf_counter()
    n_id = g.extract_skeleton(args.d[1], args.d[0], 15, True, 16)

    start_time = time.perf_counter()
    n_edge_index, n_edge_weight = g.reconstruct_reweighted_edge(n_id)
    n_x, cnt_x = reconstruct_x(x, n_id)
    n_y = mapping_label(y, n_id)

    n_train_mask = mapping_mask(train_mask, n_id)
    n_valid_mask = mapping_mask(valid_mask, n_id)
    n_test_mask = mapping_mask(test_mask, n_id)
    n_star = mapping_mask(star, n_id)

    num_node2 = np.max(n_edge_index) + 1
    n_star32 = np.full((num_node2,), False, dtype=np.bool_)
    n_star32[n_star] = True

    graph_skeleton.init()
    g2 = graph_skeleton.Graph(n_edge_index, n_star32)
    corr_mask = g2.get_corr_mask(1, 2)
    nt = g2.nearest_target()

    g2.drop_corr()
    n_id2 = g2.extract_skeleton(1, 2, 2, True, 1)

    merge_ratio = 0.7 # corr节点feature的占比
    x_corr = np.zeros_like(n_x)
    num_corr = np.zeros(x_corr.shape[0])
    x2 = np.zeros_like(n_x)
    for i in range(num_node2):
        if corr_mask[i]:
            x_corr[ nt[i] ] += n_x[i]
            num_corr[ nt[i] ] += 1

    num_corr = np.clip(
        num_corr,
        a_min = 1,
        a_max = None,
        )

    for i in range(num_node2):
        if n_star32[i]:
            x2[i] = (merge_ratio * n_x[i] + (1-merge_ratio) * x_corr[i]/num_corr[i])

    n_x2, cnt_x2 = reconstruct_x(x2, n_id2)

    n_edge_index2, n_edge_weight2 = g2.reconstruct_reweighted_edge(n_id2)
    print(f"Done! [{time.perf_counter() - start_time:.2f}s]")

    n_y2 = mapping_label(n_y, n_id2)
    n_train_mask2 = mapping_mask(n_train_mask, n_id2)
    n_valid_mask2 = mapping_mask(n_valid_mask, n_id2)
    n_test_mask2 = mapping_mask(n_test_mask, n_id2)
    n_star2 = mapping_mask(n_star, n_id2)

    print('-'*20)
    print(f'| Skeleton-gamma | #V: {n_x2.shape[0]} | #E: {n_edge_index2.shape[1]} | #Target: {n_star2.shape[0]} |')
    print('BCR: {:.3f}'.format(1-(x.shape[0]-n_x2.shape[0])/(x.shape[0]-star.shape[0])))

    skeleton_data = { 'x': n_x2, 'y': n_y2, 
                'edge_index': n_edge_index2, 
                'edge_weight': n_edge_weight2, 
                'train_mask': n_train_mask2, 'valid_mask': n_valid_mask2, 'test_mask': n_test_mask2}
    np.save(root + '/skeleton_gamma.npy', skeleton_data)


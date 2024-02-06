import numpy as np
import torch
from torch_geometric.utils import to_undirected
import os
from boxprint import bprint
from torch_geometric.utils import to_undirected
import torch
import torch.nn.functional as F
import time
from torch_geometric.data import Data
import numpy as np


def dgraph_cut():

    print('no data exists, cutting edges... ')
    root = '../../datasets/DGraphFin'
    file_path = root + '/dgraphfin.npz'
    save_path = root+'/edge_cut'
    dataset = np.load(file_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_mask = dataset['train_mask']
    valid_mask = dataset['valid_mask']
    test_mask = dataset['test_mask']

    x = dataset['x']
    y = dataset['y']
    edge_index = dataset['edge_index']
    edge_attr = None

    all_mask = np.array(list(range(len(x))))
    target_mask = np.concatenate((train_mask, valid_mask, test_mask), axis = 0)

    a = np.full( (len(x),), True, dtype=bool)
    a[target_mask] = False
    target_mask_bool = a

    cut_ratio = 1
    bg_mask = all_mask[target_mask_bool]
    np.random.shuffle(bg_mask)
    bg_cut_mask = bg_mask[:int(cut_ratio*len(bg_mask))]

    a = np.full( (len(x),), True, dtype=bool)
    a[bg_cut_mask] = False
    bg_mask_bool = a

    print('target_mask lenght', len(target_mask))
    print('bg_mask lenght', len(bg_mask))
    print('bg_cut_mask lenght', len(bg_cut_mask))

    # cut T-T
    print("---------- cut T-T ----------")
    tg_adj_list = []
    for edge in edge_index:
        if not target_mask_bool[edge[0]] and not target_mask_bool[edge[1]]:
            continue
        else:
            tg_adj_list.append(edge)

    print('cut ratio: {:.3f}, edge lenght {}'.format(1 - len(tg_adj_list)/len(edge_index), len(tg_adj_list)))
    np.save(save_path + '/cut_tt.npy', np.array(tg_adj_list))

    # cut B-B
    print("---------- cut B-B ----------")
    bg_adj_list = []
    for edge in edge_index:
        if not bg_mask_bool[edge[0]] and not bg_mask_bool[edge[1]]:
            continue
        else:
            bg_adj_list.append(edge)

    print('cut ratio: {:.3f}, edge lenght {}'.format(1 - len(bg_adj_list)/len(edge_index), len(bg_adj_list)))
    np.save(save_path + '/cut_bb.npy', np.array(bg_adj_list))


    # cut T-B
    print("---------- cut T-B ----------")
    tb_adj_list = []
    for edge in edge_index:
        if not target_mask_bool[edge[0]] and not target_mask_bool[edge[1]]:
            tb_adj_list.append(edge)

        if not bg_mask_bool[edge[0]] and not bg_mask_bool[edge[1]]:
            tb_adj_list.append(edge)

    print('cut ratio: {:.3f}, edge lenght {}'.format(1 - len(tb_adj_list)/len(edge_index), len(tb_adj_list)))
    np.save(save_path + '/cut_tb.npy', np.array(tb_adj_list))

    # mask for random cut
    print("---------- cut randomly ----------")
    random_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    for random_ratio in random_ratios:
        rd_mask = np.random.choice(len(edge_index),int((1-random_ratio)*len(edge_index)),replace=False)
        print('cut ratio: {}, edge lenght {}'.format(random_ratio, len(rd_mask)))

        rd_adj_list = edge_index[rd_mask]
        np.save(save_path + '/cut_random_{}.npy'.format(random_ratio), np.array(rd_adj_list))   
  

def dataloader(dataset, cut, rd_ratio):

    if dataset == 'dgraph':
        bprint("DGraph-Fin", width=20)
        root = '../../datasets/DGraphFin'
        file_path = root + '/dgraphfin.npz'
        dataset = np.load(file_path)
        
        train_mask = torch.from_numpy(dataset['train_mask'])
        valid_mask = torch.from_numpy(dataset['valid_mask'])
        test_mask = torch.from_numpy(dataset['test_mask'])
        
        x = torch.from_numpy(dataset['x'])
        x = (x-x.mean(0))/x.std(0)
        y = torch.from_numpy(dataset['y'])
        edge_index = torch.from_numpy(dataset['edge_index'].T)

        if cut in ['tt', 'bb', 'tb', 'cut_random']:
            if cut == 'tt':
                file_name = root + '/edge_cut/cut_tt.npy'
                print('------- CUT T-T -------')
            elif cut == 'bb':
                file_name = root + '/edge_cut/cut_bb.npy'
                print('------- CUT B-B -------')
            elif cut == 'tb':
                file_name = root + '/edge_cut/cut_tb.npy'
                print('------- CUT T-B -------')
            elif cut == 'cut_random':
                file_name = root + '/edge_cut/cut_random_{}.npy'.format(rd_ratio)
                print('-------CUT RANDOM-------')

            if not os.path.exists(file_name):
                dgraph_cut()
            edge_index = torch.from_numpy(np.load(file_name).T)

        print('| #N: {} | #E: {} |'.format(x.shape[0], edge_index.shape[1]))
        edge_index = to_undirected(edge_index)
        edge_attr = None
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    return data


# if __name__ == '__main__':
#     # arxiv_cut()
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


def dataloader(cut):

    bprint("DGraph-Fin", width=20)
    root = '../../datasets/DGraphFin'
    file_path = root + '/dgraphfin.npz'
    dataset = np.load(file_path)
    
    train_mask = torch.from_numpy(dataset['train_mask'])
    valid_mask = torch.from_numpy(dataset['valid_mask'])
    test_mask = torch.from_numpy(dataset['test_mask'])
    train_idx = train_mask
    
    x = torch.from_numpy(dataset['x'])
    x = (x-x.mean(0))/x.std(0)
    y = torch.from_numpy(dataset['y'])
    edge_index = torch.from_numpy(dataset['edge_index'].T)

    if cut in ['skeleton_alpha','skeleton_beta','skeleton_gamma',]:
        print('-------Use skeleton-------')
        if cut == 'skeleton_gamma':
            skeleton = np.load(root + '/skeleton/skeleton_gamma.npy', allow_pickle=True).item()
        elif cut == 'skeleton_beta':
            skeleton = np.load(root + '/skeleton/skeleton_beta.npy', allow_pickle=True).item()
        elif cut == 'skeleton_alpha':
            skeleton = np.load(root + '/skeleton/skeleton_alpha.npy', allow_pickle=True).item()
        train_mask = torch.from_numpy(skeleton['train_mask'])
        valid_mask = torch.from_numpy(skeleton['valid_mask'])
        test_mask = torch.from_numpy(skeleton['test_mask'])
        # print('train_mask:', train_mask)

        x = torch.from_numpy(skeleton['x'])
        x = (x-x.mean(0))/x.std(0)
        y = torch.from_numpy(skeleton['y'])
        edge_index = torch.from_numpy(skeleton['edge_index']).type(torch.int64)

    print('| #N: {} | #E: {} |'.format(x.shape[0], edge_index.shape[1]))
    edge_index = to_undirected(edge_index)
    edge_attr = None
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    return data


# if __name__ == '__main__':
#     # arxiv_cut()
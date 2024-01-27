from torch_geometric.utils import to_undirected
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.data import Data
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch_geometric.transforms as T
import numpy as np
from sklearn import metrics
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
from models import GCN
from config import config, update_config


def train(model, data, train_idx, optimizer, no_conv=False):
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, masks, no_conv=False):
    model.eval()

    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)
    y_pred = out.exp()

    aucs = []
    for mask in masks:
        test_auc = metrics.roc_auc_score(data.y[mask].cpu().numpy(), y_pred[mask,1].detach().cpu().numpy())
        aucs.append(test_auc)
    return aucs


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
# -----------------------------------------------general settings--------------------------------------------------
parser.add_argument('--device', default='cpu',
                    help='device for computing')
parser.add_argument('--cut', default='no', choices = ['no', 'tg', 'bg', 'skeleton_alpha','skeleton_beta','skeleton_gamma'],
                    help='cut target or random')
parser.add_argument('--model', default='gcn', choices = ['gcn'],
                    help='model')
parser.add_argument('--sel_ratio', type=float, default=0.5)
parser.add_argument('--rd_ratio', default=0.1, type=float,
                    help='random cut ratio')
parser.add_argument('--k', default=2, type=int,
                    help='hop number for skeleton')
parser.add_argument('--mlp', action='store_true',
                    help='if use mlp for classification')   
parser.add_argument('--sample-mask', action='store_true',
                    help='sample using mask?') 
parser.add_argument('--iter', default=10, type=int, 
                    help='iteration for running')  
args = parser.parse_args()
update_config(config, args.model)
print(args)


EPS = 1e-15
if args.device == 'cpu':
    print('cpu')
    device = torch.device('cpu')
else:
    print('gpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print('Using GPU:' + str(np.argmin(memory_gpu)))
    device = torch.device('cuda:'+str(np.argmin(memory_gpu)))


if args.cut != 'skeleton_gamma':
    file_path = '/home/cao.1378/project/dataset/dgraph/dgraphfin.npz'
    dataset = np.load(file_path)
    train_mask = torch.from_numpy(dataset['train_mask'])
    valid_mask = torch.from_numpy(dataset['valid_mask'])
    test_mask = torch.from_numpy(dataset['test_mask'])
    print('train_mask:', train_mask)
    train_idx = train_mask
    x = torch.from_numpy(dataset['x'])
    x = (x-x.mean(0))/x.std(0)
    y = torch.from_numpy(dataset['y']).to(device)
    print(x.shape)
    if args.cut == 'tg':
        edge_index = torch.from_numpy(np.load('/home/zhangboning/clf/data/xinye_cut/cut_tg.npy').T)
        print('-------CUT TARGET-------')
    elif args.cut == 'bg':
        edge_index = torch.from_numpy(np.load('/home/zhangboning/clf/data/xinye_cut/cut_bg.npy').T)
        print('-------CUT BACKGROUND-------')
    elif args.cut == 'cut_random':
        edge_index = torch.from_numpy(np.load('/home/zhangboning/clf/data/xinye_cut/cut_random_{}.npy'.format(args.rd_ratio)).T)
        print('-------CUT RANDOM-------')
    elif args.cut == 'random':
        # edge_index = torch.from_numpy(np.load('/home/caolinfeng/clf/data/xinye_cut/rd_select_bg_{}.npy'.format(args.sel_ratio)).T)
        edge_index = torch.from_numpy(np.load('/home/cao.1378/project/dataset/dgraph/coreset/random_{}.npy'.format(args.sel_ratio)).T)
        print('edge:',edge_index.shape)
        print('-------CUT RANDOM BACKGROUDNS-------')
    elif args.cut == 'cent_p':
        edge_index = torch.from_numpy(np.load('/home/cao.1378/project/dataset/dgraph/coreset/Cent_P_{}.npy'.format(args.sel_ratio)).T)
        print('edge:',edge_index.shape)
        print('-------Central Rank (Page Rank)-------')
    elif args.cut == 'cent_d':
        edge_index = torch.from_numpy(np.load('/home/cao.1378/project/dataset/dgraph/coreset/Cent_D_{}.npy'.format(args.sel_ratio)).T)
        print('edge:',edge_index.shape)
        print('-------Central Rank (Degree)-------')
    elif args.cut == 'ac':
        k = args.k
        edge_index = torch.from_numpy(np.load('/home/zhangboning/clf/data/xinye_skelecton/{}/edge_skeleton_relabel.npy'.format(k)).T)
        node_id = torch.from_numpy(np.load('/home/zhangboning/clf/data/xinye_skelecton/{}/relabel_id.npy'.format(k)))
        train_mask = torch.from_numpy(np.load('/home/zhangboning/clf/data/xinye_skelecton/{}/relabel_train.npy'.format(k)))
        valid_mask = torch.from_numpy(np.load('/home/zhangboning/clf/data/xinye_skelecton/{}/relabel_valid.npy'.format(k)))
        test_mask = torch.from_numpy(np.load('/home/zhangboning/clf/data/xinye_skelecton/{}/relabel_test.npy'.format(k)))
        x = x[node_id]
        y = y[node_id]
        print('-------USE ANCHOR-------')
        print(x.shape)
        print('k:', k)
    elif args.cut == 'no':
        edge_index = torch.from_numpy(dataset['edge_index'].T)
        print('-------NO CUT-------')

elif args.cut == 'skeleton_gamma':
    print('-------skeleton-------')
    # zip = np.load('/home/caolinfeng/clf/data/xinye_cut/zip_graph_[1, 1, 1].npy', allow_pickle=True).item()
    # zip = np.load('/home/caolinfeng/clf/data/xinye_cut/zip0622_graph_[1, 2, 2].npy', allow_pickle=True).item()
    zip = np.load('/home/cao.1378/project/dataset/dgraph/skeleton/zip_merge_graph_[1, 2, 15]_0.7.npy', allow_pickle=True).item()
    train_mask = torch.from_numpy(zip['train_mask'])
    valid_mask = torch.from_numpy(zip['valid_mask'])
    test_mask = torch.from_numpy(zip['test_mask'])
    print('train_mask:', train_mask)
    train_idx = train_mask

    x = torch.from_numpy(zip['x'])
    x = (x-x.mean(0))/x.std(0)
    y = torch.from_numpy(zip['y']).to(device)
    edge_index = torch.from_numpy(zip['edge_index']).type(torch.int64)

elif args.cut == 'schur_c':
    print('-------schur_c-------')
    zip = np.load('/home/caolinfeng/clf/data/xinye_cut/schur_c/remap.npy', allow_pickle=True).item()
    train_mask = torch.from_numpy(zip['train_mask'])
    valid_mask = torch.from_numpy(zip['valid_mask'])
    test_mask = torch.from_numpy(zip['test_mask'])
    print('train_mask:', train_mask)
    train_idx = train_mask

    edge_index = torch.from_numpy(np.load('/home/caolinfeng/clf/data/xinye_cut/schur_c/schur_edge_0.6.npy')).type(torch.int64)
    x = torch.from_numpy(zip['x'])
    x = (x-x.mean(0))/x.std(0)
    y = torch.from_numpy(zip['y']).to(device)

edge_index = to_undirected(edge_index)
print('#N: {}, #E: {}'.format(x.shape, edge_index.shape))

edge_attr = None
data = Data(x=x.float(), edge_index=edge_index, edge_attr=edge_attr, y=y)
transform=T.ToSparseTensor()
data = transform(data)
data.adj_t = data.adj_t.to_symmetric()

data = data.to(device)
train_idx = train_idx.to(device)

if args.model == 'gcn':
    model = GCN(in_channels = data.num_features, 
                hidden_channels = config['hidden_channels'],
                out_channels = 2, 
                num_layers = config['num_layers'],
                dropout = config['dropout'],
                ).to(device)

print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

best_val_auc = best_val_auc = 0
val_aucs = []
Time = []
for run in range(args.iter):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    start = time.perf_counter()
    best_val_auc = final_val_auc = 0
    for epoch in range(config['epochs']):
        loss = train(model, data, train_idx, optimizer, no_conv=False)
        
        if epoch % 5 == 0:
            print(f'Train Epoch {epoch:02d}, Loss: {loss:.4f}')
            val_auc, val_auc = test(model, data, [valid_mask, test_mask])
            print(f'Evaluation Epoch:{epoch:02d}: Val: {val_auc:.4f}, '
                  f'Test: {val_auc:.4f}')

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_val_auc = val_auc
    val_aucs.append(final_val_auc)
    end = time.perf_counter()
    Time.append(end-start)
    print('-----------------------')
    print('Consuming time{}:'.format(end-start))
    print('best acc{}:'.format(final_val_auc))

val_auc = torch.tensor(val_aucs)
aver_time = torch.tensor(Time)
print('============================')
print(f'Final Test: {val_auc.mean():.4f} ± {val_auc.std():.4f}')
print(f'aver time: {aver_time.mean():.4f} ± {aver_time.std():.4f}')
print(val_aucs)
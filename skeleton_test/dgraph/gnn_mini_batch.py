import os
os.chdir('/home/cao.1378/project/skeleton_betaaseline/basemodel/dgraph/model') 

from torch_geometric.utils import to_undirected
import torch
import torch.nn.functional as F
import time
from torch_geometric.data import Data
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.loader import NeighborSampler
import numpy as np
from sklearn import metrics
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models import GATv2_NeighSampler, SAGE_NeighSampler, GIN_NeighSampler
from config import config, update_config


def train():
    model.train()

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        n_id = n_id.to(device)

        optimizer.zero_grad()
        out = model(x[n_id], adjs)

        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)
    return loss, approx_acc


@torch.no_grad()
def test(masks):
    model.eval()
    out = model.inference(x)

    aucs = []
    for mask in masks:
        test_auc = metrics.roc_auc_score(data.y[mask].cpu().numpy(), out[mask,1].detach().cpu().numpy())
        aucs.append(test_auc)
    return aucs


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
# -----------------------------------------------general settings--------------------------------------------------
parser.add_argument('--device', default='gpu',
                    help='device for computing')
parser.add_argument('--cut', default='no', choices = ['no', 'tg', 'bg', 'tb', 'skeleton_alpha','skeleton_beta','skeleton_gamma'],
                    help='cut target or random')
parser.add_argument('--model', default='sage', choices = ['sage', 'gat', 'gin'],
                    help='model')
parser.add_argument('--rd_ratio', default=0.1, type=float,
                    help='random cut ratio')
parser.add_argument('--mlp', action='store_true',
                    help='if use mlp for classification')
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


if args.cut not in ['skeleton_alpha','skeleton_beta','skeleton_gamma',]:
    file_path = '/home/cao.1378/project/dataset/dgraph/dgraphfin.npz'
    dataset = np.load(file_path)
    train_mask = torch.from_numpy(dataset['train_mask'])
    valid_mask = torch.from_numpy(dataset['valid_mask'])
    test_mask = torch.from_numpy(dataset['test_mask'])
    train_idx = train_mask
    x = torch.from_numpy(dataset['x'])
    x = (x-x.mean(0))/x.std(0)
    y = torch.from_numpy(dataset['y']).to(device)
    print(x.shape)
    if args.cut == 'tg':
        edge_index = torch.from_numpy(np.load('/home/cao.1378/project/dataset/dgraph/edge_cut/cut_tg.npy').T)
        print('------- CUT T-T -------')
    elif args.cut == 'bg':
        edge_index = torch.from_numpy(np.load('/home/cao.1378/project/dataset/dgraph/edge_cut/cut_bg.npy').T)
        print('------- CUT B-B -------')
    elif args.cut == 'tb':
        edge_index = torch.from_numpy(np.load('/home/cao.1378/project/dataset/dgraph/edge_cut/cut_tb.npy').T)
        print('------- CUT T-B -------')
    elif args.cut == 'cut_random':
        edge_index = torch.from_numpy(np.load('/home/cao.1378/project/dataset/dgraph/edge_cut/cut_random_{}.npy'.format(args.rd_ratio)).T)
        print('-------CUT RANDOM-------')


elif args.cut in ['skeleton_alpha','skeleton_beta','skeleton_gamma',]:
    print('-------skeleton-------')
    if args.cut == 'skeleton_gamma':
        zip = np.load('/home/cao.1378/project/dataset/dgraph/skeleton/zip_merge_graph_[1, 2, 15]_0.7.npy', allow_pickle=True).item()
    elif args.cut == 'skeleton_beta':
        zip = np.load('/home/cao.1378/project/dataset/dgraph/skeleton/zip0622_graph_[1, 2, 20].npy', allow_pickle=True).item()
    elif args.cut == 'skeleton_alpha':
        zip = np.load('/home/cao.1378/project/dataset/dgraph/skeleton/zip_graph_[1, 2, 2].npy', allow_pickle=True).item()
    train_mask = torch.from_numpy(zip['train_mask'])
    valid_mask = torch.from_numpy(zip['valid_mask'])
    test_mask = torch.from_numpy(zip['test_mask'])
    print('train_mask:', train_mask)
    train_idx = train_mask

    x = torch.from_numpy(zip['x'])
    x = (x-x.mean(0))/x.std(0)
    y = torch.from_numpy(zip['y']).to(device)
    edge_index = torch.from_numpy(zip['edge_index']).type(torch.int64)

edge_index = to_undirected(edge_index)
print('#N: {}, #E: {}'.format(x.shape, edge_index.shape))
edge_attr = None
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

train_loader = NeighborSampler(edge_index, node_idx=train_mask,
                               sizes=config['sample_size'], batch_size=config['batch_size'],
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(edge_index, node_idx=None, sizes=[-1],
                                  batch_size=config['batch_size'], shuffle=False,
                                  num_workers=12)

if args.model == 'sage':
    model = SAGE_NeighSampler(device, subgraph_loader, x.shape[1], 
                                config['hidden_channels'], 
                                out_channels=2, 
                                num_layers=config['num_layers'], 
                                dropout=config['dropout'], 
                                batchnorm=config['batchnorm'])

elif args.model == 'gat':
    model = GATv2_NeighSampler(device, subgraph_loader, x.shape[1], 
                                config['hidden_channels'], 
                                out_channels=2, 
                                num_layers=config['num_layers'], 
                                dropout=config['dropout'], 
                                layer_heads=[4,2,1], 
                                batchnorm=config['batchnorm'])
elif args.model == 'gin':
    model = GIN_NeighSampler(device, subgraph_loader, x.shape[1], 
                                config['hidden_channels'], 
                                out_channels=2, 
                                num_layers=config['num_layers'], 
                                dropout=config['dropout'], 
                                batchnorm=config['batchnorm'])

print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
x, edge_index = x.to(torch.float32).to(device), edge_index.to(torch.float32).to(device)

best_val_acc = best_test_acc = 0
test_accs = []
Time = []
Log_train = []
Log_valid = []
Log_test = []
for run in range(args.iter):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    start = time.perf_counter()
    best_val_acc = final_test_acc = 0
    log_train = []
    log_valid = []
    log_test = []
    for epoch in range(config['epochs']):
        loss, acc = train()
        if epoch % 1 == 0:
            print(f'Train Epoch {epoch:02d}, Loss: {loss:.4f}')
            val_acc, test_acc = test([valid_mask, test_mask])
            print(f'Evaluation Epoch:{epoch:02d}: Val: {val_acc:.4f},'
                  f'Test: {test_acc:.4f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            
            log_train.append(acc)
            log_valid.append(val_acc)
            log_test.append(test_acc)

    test_accs.append(final_test_acc)
    end = time.perf_counter()
    Time.append(end-start)
    Log_train.append(log_train)
    Log_valid.append(log_valid)
    Log_test.append(log_test)
    print('-----------------------')
    print('Consuming time{}:'.format(end-start))
    print('best acc{}:'.format(final_test_acc))

test_acc = torch.tensor(test_accs)
aver_time = torch.tensor(Time)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
print(f'aver time: {aver_time.mean():.4f} ± {aver_time.std():.4f}')
print(test_accs)
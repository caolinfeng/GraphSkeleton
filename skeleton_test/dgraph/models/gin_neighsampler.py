from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn import ReLU, Sequential
from torch_geometric.nn import GINConv

class GIN_NeighSampler(torch.nn.Module):
    def __init__(self
                 ,device, subgraph_loader
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(GIN_NeighSampler, self).__init__()

        self.layer_loader = subgraph_loader
        self.device = device
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.dropout = dropout

        for i in range(num_layers):
            mlp = Sequential(
                Lin(in_channels, 2 * hidden_channels),
                torch.nn.BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Lin(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True).jittable()

            self.convs.append(conv)

            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
            in_channels = hidden_channels

        self.lin1 = Lin(hidden_channels, hidden_channels)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin2 = Lin(hidden_channels, out_channels)

        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.batch_norm1.reset_parameters()     
        
        
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        x = x[:size[1]]

            # x = self.convs[i]((x, x_target), edge_index)
            # if i != self.num_layers-1:
            #     if self.batchnorm:
            #         x = self.bns[i](x)
            #     x = F.relu(x)
            #     x = F.dropout(x, p=0.5, training=self.training)
                
        return x.log_softmax(dim=-1)
    
    '''
    subgraph_loader: size = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=**, shuffle=False,
                                  num_workers=12)
    You can also sample the complete k-hop neighborhood, but this is rather expensive (especially for Reddit). 
    We apply here trick here to compute the node embeddings efficiently: 
       Instead of sampling multiple layers for a mini-batch, we instead compute the node embeddings layer-wise. 
       Doing this exactly k times mimics a k-layer GNN.  
    '''
    
    def inference_all(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
    def inference(self, x_all):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in self.layer_loader:
                edge_index, _, size = adj.to(self.device)
                x = x_all[n_id].to(self.device)
                x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
                x = x[:size[1]]
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
        
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x_all.log_softmax(dim=-1)

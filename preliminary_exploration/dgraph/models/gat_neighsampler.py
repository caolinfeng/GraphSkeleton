from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv
from tqdm import tqdm

class GAT_NeighSampler(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , layer_heads = []
                 , batchnorm=True):
        super(GAT_NeighSampler, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        
        if len(layer_heads)>1:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[0]))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels*layer_heads[i-1], hidden_channels, heads=layer_heads[i], concat=True))
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[i-1]))
            self.convs.append(GATConv(hidden_channels*layer_heads[num_layers-2]
                              , out_channels
                              , heads=layer_heads[num_layers-1]
                              , concat=False))
        else:
            self.convs.append(GATConv(in_channels, out_channels, heads=layer_heads[0], concat=False))        

        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()        
        
        
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                
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
    
    def inference(self, x_all, layer_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm: 
                        x = self.bns[i](x)
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all.log_softmax(dim=-1)



class GATv2_NeighSampler(torch.nn.Module):
    def __init__(self,
                device, subgraph_loader, 
                in_channels,
                hidden_channels,
                out_channels,
                num_layers,
                dropout,
                layer_heads = [],
                batchnorm=True):
        super(GATv2_NeighSampler, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        self.device = device
        self.subgraph_loader = subgraph_loader
        
        if len(layer_heads)>1:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[0]))
            for i in range(1, num_layers - 1):
                self.convs.append(GATv2Conv(hidden_channels*layer_heads[i-1], hidden_channels, heads=layer_heads[i], concat=True))
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[i-1]))
            self.convs.append(GATv2Conv(hidden_channels*layer_heads[num_layers-2]
                              , out_channels
                              , heads=layer_heads[num_layers-1]
                              , concat=False))
        else:
            self.convs.append(GATv2Conv(in_channels, out_channels, heads=layer_heads[0], concat=False))        

        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()        
        
        
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
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
        pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in self.subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x = x_all[n_id].to(self.device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm: 
                        x = self.bns[i](x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, args, device, subgraph_loader, in_channels, hidden_channels, out_channels, num_layers, batchnorm=True):
        super().__init__()

        self.num_layers = num_layers
        self.args = args
        self.device = device
        self.subgraph_loader = subgraph_loader

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        if args.mlp:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels//2))
            self.linear = nn.Sequential(
                nn.Linear(hidden_channels//2, hidden_channels//4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels//4, hidden_channels//8),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels//8, out_channels)
            )
        else:
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()        
        

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x.to(torch.float32), x_target.to(torch.float32)), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        if self.args.mlp:
            x = self.linear(x)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in self.subgraph_loader:
                # print(batch_size)
                edge_index, _, size = adj.to(self.device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(self.device)
                x_target = x[:size[1]]
                x = self.convs[i]((x.to(torch.float32), x_target.to(torch.float32)), edge_index)
                # print(x.shape)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm: 
                        x = self.bns[i](x)

                if i == self.num_layers - 1:
                    if self.args.mlp:
                        x = self.linear(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all.log_softmax(dim=-1)


    def inference_all(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
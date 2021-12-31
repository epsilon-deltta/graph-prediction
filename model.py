from gnn import GNN

import torch
from torch_geometric import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch.nn.functional as F

class GCN_pyg(torch.nn.Module):
    def __init__(self, num_layer=3, emb_dim=18, drop_ratio = 0.5, JK = "last", residual = True, graph_pooling = "mean"):
        super(GCN_pyg,self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
            
        self.convs0  = torch.nn.ModuleList()
        self.convs1  = torch.nn.ModuleList()
        self.convs2  = torch.nn.ModuleList()
        self.lins    = torch.nn.ModuleList()
        self.bond_encoder = torch.nn.ModuleList()
        
        self.encoder = AtomEncoder(emb_dim)
        
        for layer in range(num_layer):
            self.lins  .append(torch.nn.Linear(self.emb_dim,self.emb_dim))
            
            self.bond_encoder.append()
            
            self.convs0.append(nn.GCNConv(self.emb_dim,self.emb_dim) )
            self.convs1.append(nn.GCNConv(self.emb_dim,self.emb_dim) )
            self.convs2.append(nn.GCNConv(self.emb_dim,self.emb_dim) )
            # self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        fc_list =    [ torch.nn.Linear(emb_dim,emb_dim)
                      ,torch.nn.Linear(emb_dim,emb_dim)
                      ,torch.nn.Linear(emb_dim,1)
                     ]
        self.fc = torch.nn.Sequential(*fc_list)
        
    def forward(self,batched_data):
        
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        
        ### computing input node embedding

        h_list = [self.encoder(x)]
        for layer in range(self.num_layer):
            
            print(h_list[layer].shape, edge_index.shape, edge_attr[:,0].shape)
            
            h0 = self.convs0[layer](h_list[layer], edge_index, edge_attr[:,0])
            h1 = self.convs1[layer](h_list[layer], edge_index, edge_attr[:,1])
            h2 = self.convs2[layer](h_list[layer], edge_index, edge_attr[:,2])
            h  = h0 + h1 + h2
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            h_node = h_list[-1]
        elif self.JK == "sum":
            h_node = 0
            for layer in range(self.num_layer + 1):
                h_node += h_list[layer]
        
        h_graph = self.pool(h_node,batch)
        
        return self.fc(h_graph)

def model_selector(**args):
    
    args['gnn'] = args['gnn'].lower()
    
    if args['gnn'] == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = args['num_tasks'], num_layer = args['num_layer'], emb_dim = args['emb_dim'], drop_ratio = args['drop_ratio'], virtual_node = False)
    elif args['gnn'] == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = args['num_tasks'], num_layer = args['num_layer'], emb_dim = args['emb_dim'], drop_ratio = args['drop_ratio'], virtual_node = True)
    elif args['gnn'] == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = args['num_tasks'], num_layer = args['num_layer'], emb_dim = args['emb_dim'], drop_ratio = args['drop_ratio'], virtual_node = False)
    elif args['gnn'] == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = args['num_tasks'], num_layer = args['num_layer'], emb_dim = args['emb_dim'], drop_ratio = args['drop_ratio'], virtual_node = True)
    
    elif args['gnn'] == 'gcn-pyg':
        model = GCN_pyg()
    elif args['gnn'] == 'gat-pyg':
        pass
    elif args['gnn'] == 'transformerconv':
        pass
    elif args['gnn'] == 'hig':
        pass
    else:
        raise ValueError('Invalid GNN type')
        
    return model


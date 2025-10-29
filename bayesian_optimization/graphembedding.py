from gpytorch.kernels import Kernel,RBFKernel,ScaleKernel
import torch
from torch import nn
from torch.nn import ReLU,Sequential
import torch.nn.functional as F
from collections import defaultdict, Counter
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from topomodelx.nn.combinatorial.hmc import HMC
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from topomodelx.nn.combinatorial.hmc import  HMCLayer
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm
class GraphRegressor(nn.Module):
    def __init__(self,n,nx,nz,hidden_channel = 128, embedding_mode='GIN'):
        super().__init__()
        self.embedding_mode = embedding_mode
        if self.embedding_mode in ['GIN']:
            self.embedder = GINEmbedder(n,nx,nz,hidden_channel)
        elif self.embedding_mode in ['GT']:
            self.embedder = GraphTransformerEmbedder(n, nx, nz, hidden_dim=128)
        elif self.embedding_mode in ['GCN']:
            self.embedder = GCNEmbedder(n,nx,nz,hidden_channel)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channel, 256),
            nn.BatchNorm1d(256), 
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), 
            nn.LeakyReLU(0.1),
            nn.Linear(512,1)
        ) 
    def forward(self,x):
        embedding = self.embedder(x)
        return self.fc(embedding)
class GraphEmbedderforGP(nn.Module):
    def __init__(self,n,nx,nz,hidden_channel = 128, embedding_mode='GIN'):
        super().__init__()
        self.embedding_mode = embedding_mode
        if self.embedding_mode in ['GIN']:
            self.embedder = GINEmbedder(n,nx,nz,hidden_channel)
        elif self.embedding_mode in ['GT']:
            self.embedder = GraphTransformerEmbedder(n, nx, nz, hidden_dim=hidden_channel)
        elif self.embedding_mode in ['GCN']:
            self.embedder = GCNEmbedder(n,nx,nz,hidden_channel)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channel, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
        ) 
    def forward(self,x):
        embedding = self.embedder(x)
        return self.fc(embedding)

        
class GraphEmbedder(nn.Module):
    def __init__(self, n, nx, nz):
        super().__init__()
        q = torch.zeros(n, 3);  q[:, 0] = 1.0
        x = torch.zeros(nx, 3); x[:, 1] = 1.0
        z = torch.zeros(nz, 3); z[:, 2] = 1.0
        self.register_buffer('single_nodes', torch.cat((q, x, z), dim=0))

    def batch_transform(self, adjacency_matrices):
        batch_size = adjacency_matrices.size(0)
        data_list = []
        for i in range(batch_size):
            adj = adjacency_matrices[i]
            edge_index, _ = dense_to_sparse(adj)
            data_list.append(Data(x=self.single_nodes, edge_index=edge_index))
        return Batch.from_data_list(data_list).to(adjacency_matrices.device)



from torch_geometric.nn import GINConv, global_mean_pool, GraphNorm
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
class GCNEmbedder_WithLogicalOperators(nn.Module):
    def __init__(self,hidden_dim,num_layers):
        super().__init__()
        # 1) normalize the raw 3-dim node features
        self.input_norm = nn.BatchNorm1d(5)

        # GCN layers and batch norms
        self.layers = nn.ModuleList()
        self.bns    = nn.ModuleList()
        for i in range(num_layers):
            in_dim = 5 if i == 0 else hidden_dim
            self.layers.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 2) normalize the pooled graph-embedding
        self.post_pool_norm = nn.LayerNorm(hidden_dim)

        self.num_layers = num_layers
    def forward(self,batch):
        
        x, edge_index = batch.x, batch.edge_index

        # --- input normalization ---
        x = self.input_norm(x)

        # message passing with residuals
        h = x
        for i in range(self.num_layers):
            h_new = self.layers[i](h, edge_index)
            h_new = self.bns[i](h_new)
            # simple residual: add previous features for layers > 0
            h = F.relu(h_new + (h if i > 0 else 0))

        # global mean pooling
        out = global_mean_pool(h, batch.batch)

        # --- embedding normalization ---
        out = self.post_pool_norm(out)
        return out  # [batch_size, hidden_dim]
    
class GCNEmbedder(GraphEmbedder):
    def __init__(self, n, nx, nz, hidden_dim=128, num_layers=3):
        super().__init__(n, nx, nz)
        # 1) normalize the raw 3-dim node features
        self.input_norm = nn.BatchNorm1d(3)

        # GCN layers and batch norms
        self.layers = nn.ModuleList()
        self.bns    = nn.ModuleList()
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_dim
            self.layers.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 2) normalize the pooled graph-embedding
        self.post_pool_norm = nn.LayerNorm(hidden_dim)

        self.num_layers = num_layers

    def forward(self, adjacency_matrices):
        # transform list of adjacency matrices into a torch_geometric Batch
        batch = self.batch_transform(adjacency_matrices)
        x, edge_index = batch.x, batch.edge_index

        # --- input normalization ---
        x = self.input_norm(x)

        # message passing with residuals
        h = x
        for i in range(self.num_layers):
            h_new = self.layers[i](h, edge_index)
            h_new = self.bns[i](h_new)
            # simple residual: add previous features for layers > 0
            h = F.relu(h_new + (h if i > 0 else 0))

        # global mean pooling
        out = global_mean_pool(h, batch.batch)

        # --- embedding normalization ---
        out = self.post_pool_norm(out)
        return out  # [batch_size, hidden_dim]

class GINEmbedder(GraphEmbedder):
    def __init__(self, n, nx, nz, hidden_dim=128, num_layers=3):
        super().__init__(n, nx, nz)
        # 1) normalize the raw 3‐dim node features
        self.input_norm = nn.BatchNorm1d(3)

        self.layers = nn.ModuleList()
        self.bns    = nn.ModuleList()
        for i in range(num_layers):
            in_dim = 3 if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                # ReLU for first layer, LeakyReLU thereafter
                nn.ReLU() if i == 0 else nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers.append(GINConv(mlp))
            # you could swap to GraphNorm if you want normalization per-graph:
            # self.bns.append(GraphNorm(hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 2) normalize the pooled graph‐embedding
        self.post_pool_norm = nn.LayerNorm(hidden_dim)

        self.num_layers = num_layers

    def forward(self, adjacency_matrices):
        batch = self.batch_transform(adjacency_matrices)
        x, edge_index = batch.x, batch.edge_index

        # --- input normalization ---
        x = self.input_norm(x)

        # message passing with residuals
        h = x
        for i in range(self.num_layers):
            h_new = self.layers[i](h, edge_index)
            # BatchNorm1d expects shape [N_nodes, hidden_dim]
            h_new = self.bns[i](h_new)
            # simple residual + activation
            h = F.relu(h_new + (h if i > 0 else 0))

        # global mean pool
        out = global_mean_pool(h, batch.batch)
        # --- embedding normalization ---
        out = self.post_pool_norm(out)
        return out  # [batch_size, hidden_dim]

    


class GraphTransformerEmbedder(GraphEmbedder):
    def __init__(self, n, nx, nz,
                 hidden_dim=128, num_layers=3,
                 heads=4, dropout=0.1):
        super().__init__(n, nx, nz)

        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.heads       = heads

        # ——— Project input features (3 dims) up to hidden_dim ———
        in_dim = self.single_nodes.size(1)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # ——— Build TransformerConv layers ———
        self.convs = nn.ModuleList()
        # first layer: hidden_dim → hidden_dim
        self.convs.append(
            TransformerConv(hidden_dim,
                            hidden_dim // heads,
                            heads=heads,
                            dropout=dropout)
        )
        # remaining layers
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(hidden_dim,
                                hidden_dim // heads,
                                heads=heads,
                                dropout=dropout)
            )

        # ——— per-layer feed-forward + LayerNorm ———
        self.ffn         = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, adjacency_matrices):
        # 1) batchify
        batch      = self.batch_transform(adjacency_matrices)
        x, edge_ix = batch.x, batch.edge_index

        # 2) initial projection
        h = self.input_proj(x)

        # 3) stacked Transformer layers with residuals
        for i, conv in enumerate(self.convs):
            h_res = h
            h     = conv(h, edge_ix)
            h     = F.dropout(h, p=self.dropout, training=self.training)
            h     = h + self.ffn[i](h)    # FFN residual
            h     = h + h_res             # skip residual
            h     = self.layer_norms[i](h)
            h     = F.relu(h)

        # 4) global readout
        out = global_mean_pool(h, batch.batch)
        return out  # [batch_size, hidden_dim]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool

class eGraphTransformerEmbedder(GraphEmbedder):
    def __init__(self, n, nx, nz,
                 hidden_dim=512, num_layers=6,
                 heads=8, dropout=0.1):
        super().__init__(n, nx, nz)

        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.heads       = heads


        self.single_nodes = torch.randn(n, nx)  
        in_dim = self.single_nodes.size(1)


        self.input_proj = nn.Linear(in_dim, hidden_dim)


        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

 
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 全局池化后拼接 -> 最终投影
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


    def forward(self, adjacency_matrices):
        batch = self.batch_transform(adjacency_matrices)
        x, edge_ix = batch.x, batch.edge_index
        h = self.input_proj(x)

        for i in range(self.num_layers):
            h_res = h
            h     = self.convs[i](h, edge_ix)
            h     = F.dropout(h, p=self.dropout, training=self.training)
            h     = h + self.ffn[i](h)    # FFN residual
            h     = h + h_res             # skip residual
            h     = self.layer_norms[i](h)
            h     = F.relu(h)


        mean_pool = global_mean_pool(h, batch.batch)
        max_pool  = global_max_pool(h, batch.batch)
        sum_pool  = global_add_pool(h, batch.batch)
        pooled = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)

        out = self.global_proj(pooled)  # shape: [batch_size, hidden_dim]
        return out

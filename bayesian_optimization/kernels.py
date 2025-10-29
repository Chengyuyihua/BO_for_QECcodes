"""
    This file defines the kernels used in the Bayesian Optimization algorithm
"""

from gpytorch.kernels import Kernel,RBFKernel,ScaleKernel
import torch
from torch import nn
from torch.nn import ReLU,Sequential
import torch.nn.functional as F
from collections import defaultdict, Counter
import networkx as nx

from torch_geometric.nn import GCNConv,global_mean_pool
from topomodelx.nn.combinatorial.hmc import HMC
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch

from topomodelx.nn.combinatorial.hmc import  HMCLayer



class GNNEmbedding(nn.Module):
    """
        Graph Neural Network based message passing layer (returns a cochain) for the CSS kernel
    """
    def __init__(self, n, nx, nz, out_channels=128):
        super().__init__()

        # Initial one-hot features for q, x, z nodes
        q = torch.zeros(n, 3)
        q[:, 0] = 1.0
        x = torch.zeros(nx, 3)
        x[:, 1] = 1.0
        z = torch.zeros(nz, 3)
        z[:, 2] = 1.0
        self.single_nodes = torch.cat((q, x, z), dim=0)
        

        # Graph convolutional layers
        self.conv1 = GCNConv(3, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, out_channels)

        # Fully connected layers after GNN pooling
        # First FC reduces from out_channels to out_channels // 2
        self.fc1 = torch.nn.Linear(out_channels, out_channels // 2)
        # Second FC reduces further
        self.fc2 = torch.nn.Linear(out_channels // 2, out_channels // 4)
        # Third FC outputs final embedding of desired size
        self.fc3 = torch.nn.Linear(out_channels // 4, out_channels // 8)

    def forward(self, adjacency_matrices):
        """
        adjacency_matrices: Tensor of shape [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = adjacency_matrices.size()

        # Convert each adjacency to edge-index format and build batch
        data_list = []
        for i in range(batch_size):
            adj = adjacency_matrices[i]
            edge_index, _ = dense_to_sparse(adj)
            data = Data(x=self.single_nodes.clone(), edge_index=edge_index)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)

        # GNN message passing
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.dropout(x, p=0.25, training=self.training)

        x = F.relu(self.conv2(x, batch.edge_index))
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv3(x, batch.edge_index)

        # Global pooling to get [batch_size, out_channels]
        x = global_mean_pool(x, batch.batch)
        # Normalize to unit length
        x = F.normalize(x, p=2, dim=-1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Optionally normalize final embedding
        x = F.normalize(x, p=2, dim=-1)

        return x
class GNNRegressor(nn.Module):
    def __init__(self, n, nx, nz, out_channels=128):
        super().__init__()
        self.embedding = GNNEmbedding(n, nx, nz, out_channels)
        self.fc = torch.nn.Linear(out_channels // 8, 1)
    def forward(self,adjacency_matrix):
        embedding = self.embedding(adjacency_matrix)
        out = self.fc(F.relu(embedding))
        return out
class GNNKernel(Kernel):
    def __init__(self,n,nx,nz,**kwargs):
        super().__init__(**kwargs)
        self.gnn = GNNEmbedding(n,nx,nz)
        self.rbf = RBFKernel()
        self.scale = ScaleKernel(self.rbf)
    def forward(self, x1, x2,diag=False,**params):
        # x,y = self.gnn(adjacency_matrix1, adjacency_matrix2)

        x = self.gnn(x1)
        y = self.gnn(x2)
        # print(x,y)
        return self.scale(x,y,diag=diag,**params)

    
class WLSubtreeEmbedding():
    def __init__(self, n,nx,nz,num_iterations=3, **kwargs):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations
        q = torch.zeros(n,1)
        x = torch.ones(nx,1)
        z = -torch.ones(nz,1)
        self.nodes = torch.cat((q,x,z),dim=0)
    def forward(self, edge_index1, edge_index2):

        labels1 = self.apply_wl(edge_index1, self.num_iterations)
        labels2 = self.apply_wl(edge_index2, self.num_iterations)
        # print( labels1, labels2)
        # Compute kernel between these two sets of labels
        v1,v2 =  self.compute_hist(labels1, labels2)

        
        return v1,v2

    def apply_wl(self, edge_index, num_iterations):
        neighbors = defaultdict(list)
        for i, j in edge_index.t().tolist():
            # undirected graph
            neighbors[i].append(j)
            neighbors[j].append(i)  
        

        labels = self.nodes.clone()  
        for _ in range(num_iterations):
            new_labels = labels.clone()  
            for node, node_neighbors in neighbors.items():

                neighborhood_labels = [labels[node]] + [labels[n] for n in node_neighbors]
 
                sorted_labels = sorted(neighborhood_labels)
                new_label_str = '_'.join(str(lbl) for lbl in sorted_labels)

                new_labels[node] = hash(new_label_str)
            
            labels = new_labels  
        
        return labels

    def compute_hist(self, labels1, labels2):
        # Compute a simple kernel matrix based on label histograms
        # Using dot product of histograms as an example kernel computation
        hist1 = Counter(labels1.flatten().tolist())
        hist2 = Counter(labels2.flatten().tolist())
        all_labels = list(set(list(hist1.keys()) + list(hist2.keys())))
        vec1 = torch.tensor([hist1.get(label, 0) for label in all_labels], dtype=torch.float32)
        vec2 = torch.tensor([hist2.get(label, 0) for label in all_labels], dtype=torch.float32)
        
        return vec1, vec2
class WLSubtreeKernel(Kernel):
    def __init__(self, n,nx,nz,num_iterations=3,encoder = None,eps = 1e-8,**kwargs):
        super().__init__(**kwargs)
        if encoder == None:
            def encode_(x):
                return x
            self.encode = encode_
        else:
            self.encode = encoder
        self.eps = eps
        self.num_iterations = num_iterations
        self.wl = WLSubtreeEmbedding(n,nx,nz,num_iterations)
        initial_sigma_f = 1
        initial_l = 1
        self.log_sigma_f = nn.Parameter(torch.log(torch.tensor(initial_sigma_f)))
        self.log_l = nn.Parameter(torch.log(torch.tensor(initial_l)))
    def forward(self, x1, x2, **kwargs):
        # edge_indexs1 = dense_to_sparse(edge_indexs1)[0]
        # print(f'in kernel!!!')
        edge_indices1 = self.encode(x1)
        edge_indices2 = self.encode(x2)
        

        N1 = x1.shape[0]
        N2 = x2.shape[0]

        K = torch.zeros(N1, N2, device=x1.device)
        sigma_f = torch.exp(self.log_sigma_f)
        l = torch.exp(self.log_l)

        for i in range(N1):
            for j in range(N2):
                edge_index1 = dense_to_sparse(edge_indices1[i])[0]
                edge_index2 = dense_to_sparse(edge_indices2[j])[0]
                # print(edge_index1)
                v1, v2 = self.wl.forward(edge_index1, edge_index2)
                v1_norm = v1 / (v1.norm() + self.eps)
                v2_norm = v2 / (v2.norm() + self.eps)
                cosine_sim = torch.dot(v1_norm, v2_norm)
                K[i, j] = sigma_f**2 * torch.exp(- (1 - cosine_sim) / (l**2))
        return K

    

class TwoDimensionalCCNNEmbedding(nn.Module):
    def __init__(self,n,nx,nz,layers=3, output_channel=64,negative_slope=0.2):
        """
            Combinatorial Complex Neural Network based massagepassing layer (returns a cochain) for the CSS kernel
        """
        super().__init__()
        c_1 = torch.zeros(n, 3)
        c_1[:, 1] = 1.0
        c_0 = torch.zeros(nx, 3)
        c_0[:, 0] = 1.0
        c_2 = torch.zeros(nz, 3)
        c_2[:, 2] = 1.0
        self.ccnn_0 = HMC(
            16,
            negative_slope,
        )
        self.ccnn_1 = HMC(
            32,
            negative_slope,
        )
        self.ccnn_2 = HMC(
            output_channel,
            negative_slope,
        )


    
    def forward(self,relations):
        x_0, x_1, x_2 = self.ccnn_0(
            x_0,
            x_1,
            x_2,
            relations[0],
            relations[1],
            relations[2],
            relations[3],
            relations[4],
            relations[5],
            relations[6],
            relations[7],
            relations[8],
        )
        x_0 = F.dropout(x_0, p=0.5, training=self.training)
        x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_0, x_1, x_2 = self.ccnn_1(
            x_0,
            x_1,
            x_2,
            relations[0],
            relations[1],
            relations[2],
            relations[3],
            relations[4],
            relations[5],
            relations[6],
            relations[7],
            relations[8],
        )
        x_0 = F.dropout(x_0, p=0.5, training=self.training)
        x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_0, x_1, x_2 = self.ccnn_2(
            x_0,
            x_1,
            x_2,
            relations[0],
            relations[1],
            relations[2],
            relations[3],
            relations[4],
            relations[5],
            relations[6],
            relations[7],
            relations[8],
        )
        x_0 = F.dropout(x_0, p=0.5, training=self.training)
        x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_0 = torch.nanmean(x_0, dim=0)
        x_1 = torch.nanmean(x_1, dim=0)
        x_2 = torch.nanmean(x_2, dim=0)
        out = x_0+x_1+x_2
        
        return out
class TwoDimensionalCCNNKernel(Kernel):
    def __init__(self,n,nx,nz,**kwargs):
        super().__init__(**kwargs)
        self.ccnn = TwoDimensionalCCNNEmbedding(n,nx,nz)
        self.rbf = RBFKernel()
        self.scale = ScaleKernel(self.rbf)
    def forward(self, x1, x2,diag=False,**params):

        x = self.ccnn(x1)
        y = self.ccnn(x2)
        return self.scale(x,y,diag=diag,**params)


        

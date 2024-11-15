import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import global_mean_pool, MessagePassing, global_add_pool, knn_graph, radius_graph
# from torch_geometric.utils import coalesce, remove_self_loops, add_self_loops

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra
from clifford_modules.MVLinear import MVLinear
from clifford_modules.mvlayernorm import MVLayerNorm

class GeometricAlgebraLayer(nn.Module):
    def __init__(self, algebra, in_channels, out_channels):
        super(GeometricAlgebraLayer, self).__init__()
        self.algebra = algebra
        self.fc = nn.Linear(in_channels, out_channels)
    
    def forward(self, pos):
        # Example: Using a geometric product for transformation
        transformed_pos = self.algebra.geometric_product(pos, pos)  # or any custom operation
        
        # Pass through a linear layer after geometric transformation
        return self.fc(transformed_pos)


class KNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20, aggr='max', algebra=None):
        super(KNNConv, self).__init__()
        self.k = k
        self.aggr = aggr
        self.algebra = algebra
        mul_factor = 4 if self.algebra else 2 # This is important but I don't know why
        self.fc = nn.Sequential(
            nn.Linear(in_channels * mul_factor, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.algebra_layer = GeometricAlgebraLayer(algebra, in_channels, in_channels) if algebra else None
    
    def forward(self, pos, features):
        batch_size, num_points, in_channels = pos.shape
        
        # Optionally apply geometric algebra transformations
        if self.algebra:
            pos = self.algebra_layer(pos)
            print(f"New shape: {pos.shape}")
        
        # Reshape pos and features to [N, in_channels] and [N, in_channels] for distance calculation
        pos = pos.view(-1, in_channels)
        features = features.view(-1, features.size(-1))

        # Compute pairwise distances
        dists = torch.cdist(pos, pos)  # Shape: [N, N]

        # Find the k-nearest neighbors
        knn_indices = dists.topk(self.k, largest=False).indices  # Shape: [N, k]

        # Use `torch.index_select` to gather neighbor features directly
        neighbor_features = torch.index_select(features, 0, knn_indices.view(-1)).view(-1, self.k, features.size(-1))
        
        # Concatenate center and neighbor features
        expanded_features = features.unsqueeze(1).expand(-1, self.k, -1)
        edge_features = torch.cat([expanded_features, neighbor_features - expanded_features], dim=-1)

        # Apply MLP to edge features
        edge_features = self.fc(edge_features)

        # Aggregate features using max or mean pooling
        if self.aggr == 'max':
            x = torch.max(edge_features, dim=1)[0]
        elif self.aggr == 'mean':
            x = torch.mean(edge_features, dim=1)
        else:
            raise ValueError("Unsupported aggregation method")
        
        # Reshape x back to (batch_size, num_points, out_channels)
        x = x.view(batch_size, num_points, -1)
        return x


class CGNN(nn.Module):
    def __init__(self, algebra, k=20):
        super(CGNN, self).__init__()
        self.algebra = algebra
        c_param = (2**algebra.dim) if algebra else 6

        self.conv1 = KNNConv(in_channels=c_param, out_channels=64, k=k, aggr='max', algebra=algebra)
        self.conv2 = KNNConv(in_channels=64, out_channels=128, k=k, aggr='max', algebra=None)
        
        # Fully connected layers for refining the point cloud
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    
    def forward(self, pos):
        original_pos = copy.copy(pos)

        # Make mv input if required
        if self.algebra:
            pos = self.algebra.embed_grade(pos, 1)

        # Initial features can be positional info, for example, the (x, y, z) coordinates
        features = torch.cat([pos, pos], dim=-1)  # Duplicate pos as an example of features

        # KNN-based convolution layers
        x = self.conv1(pos, features)
        x = self.conv2(pos, x)
        
        # Global pooling
        x = torch.max(x, dim=0, keepdim=True)[0]        
        
        # Refinement network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        refined_points = self.fc3(x)
        
        return refined_points + original_pos



import torch
from torch_geometric.nn import DynamicEdgeConv
import torch.nn as nn

from torch_geometric.nn import EdgeConv

import torch
from torch_cluster import knn_graph

pos = torch.rand(512, 3).to('cuda')
edge_index = knn_graph(pos, k=2, batch=None, loop=False)
print(edge_index)

exit()

# Create dummy data
pos = torch.rand(512, 3).to('cuda')
# batch = torch.zeros(pos.size(0), dtype=torch.long).to('cuda')

# Minimal `DynamicEdgeConv` setup
# conv = DynamicEdgeConv(nn.Sequential(
#     nn.Linear(6, 64),
#     nn.ReLU(),
#     nn.Linear(64, 64)
# ), k=2, aggr='max')

# conv = conv.to('cuda')

conv1 = EdgeConv(nn.Sequential(
    nn.Linear(6, 64),
    nn.ReLU(),
    nn.Linear(64, 64)
), aggr='max')
conv1 = conv1.to('cuda')
edge_index = knn_graph(pos, k=2, batch=None, loop=False)
output = conv1(pos, edge_index)

# try:
#     # output = conv(pos, batch)
#     output = conv(pos, edge_index)
#     print("DynamicEdgeConv output shape:", output.shape)
# except Exception as e:
#     print(f"Error occurred in minimal example: {e}")
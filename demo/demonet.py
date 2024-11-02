import os 
import sys
import cv2
import h5py
import torch
import random
import logging
import numpy as np
import torch.nn as nn
# from clifford import Cl  # Clifford algebra package

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# PoinTr imports
from utils import misc
from tools import builder
from utils.config import cfg_from_yaml_file

# MVP Clifford Algebra, Model, Dataset  
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra
from mvp.mvp_dataset import MVPDataset
from models.GPD import (
    GPD,
    InvariantCGENN
)

'''
### TEMNPORARY PART ###
# Chamfer Distance helper function
from torch.optim import Adam
def chamfer_distance(point_cloud1, point_cloud2):
    dist1 = torch.cdist(point_cloud1, point_cloud2, p=2).min(dim=1)[0]
    dist2 = torch.cdist(point_cloud2, point_cloud1, p=2).min(dim=1)[0]
    return dist1.mean() + dist2.mean()

# Chamfer Distance for refinement training
def chamfer_distance(pred, gt):
    diff = pred.unsqueeze(1) - gt.unsqueeze(2)
    dist = torch.sqrt((diff ** 2).sum(-1))
    dist1 = dist.min(1)[0]
    dist2 = dist.min(2)[0]
    return dist1.mean() + dist2.mean()


import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv

from torch_geometric.nn import EdgeConv
from torch_cluster import knn_graph 

class RefinementNetwork(nn.Module):
    def __init__(self, k=20):
        super(RefinementNetwork, self).__init__()
        
        # EdgeConv layers for capturing local geometric features
        self.k = k  # Set k for KNN neighborhood
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ), aggr='max')
        
        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ), aggr='max')
        
        # Fully connected layers for refining the point cloud
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
    
    def forward(self, pos):
        # Flatten the input to (N, 3)
        pos = pos.view(-1, 3)  # Shape becomes (N, 3)
        
        # Create KNN graph for EdgeConv
        edge_index = knn_graph(pos, k=self.k, batch=None, loop=False)
        
        # EdgeConv layers
        x = self.conv1(pos, edge_index)
        x = self.conv2(x, edge_index)
        
        # Global pooling
        x = torch.max(x, dim=0, keepdim=True)[0]  # Shape becomes (1, 128) after pooling
        
        # Refinement network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        refined_points = self.fc3(x)
        
        # Reshape refined points back to the original shape and add to input
        refined_points = refined_points + pos.view(1, -1, 3)
        return refined_points


# class RefinementNetwork(nn.Module):
#     def __init__(self, k=3):
#         super(RefinementNetwork, self).__init__()
        
#         # Dynamic Edge Convolution layers for capturing local geometric features
#         self.conv1 = DynamicEdgeConv(nn.Sequential(
#             nn.Linear(6, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64)
#         ), k, aggr='max')
        
#         self.conv2 = DynamicEdgeConv(nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128)
#         ), k, aggr='max')
        
#         # Fully connected layers for refining the point cloud
#         self.fc1 = nn.Linear(128, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 3)
    
#     def forward(self, pos):
#         # Flatten the input to (16384, 3) and create a batch vector
#         pos = pos.view(-1, 3)  # Shape becomes (16384, 3)
#         batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)  # Shape (16384,)

#         # Dynamic Edge Convolution layers
#         try:
#             # Move to CPU
#             pos = pos.to('cpu')
#             self.conv1 = self.conv1.to('cpu')
#             self.conv2 = self.conv2.to('cpu')

#             # Run the forward pass
#             x = self.conv1(pos, batch)
#             x = self.conv2(x, batch)
#         except Exception as e:
#             print(f"Error occurred in DynamicEdgeConv: {e}")

#         # Global pooling
#         pos = pos.to('cuda')
#         x = x.to('cuda')
#         x = torch.max(x, dim=0, keepdim=True)[0]  # Shape becomes (1, 128) after pooling
        
#         # Refinement network
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         refined_points = self.fc3(x)
        
#         # Reshape refined points back to the original shape and add to input
#         refined_points = refined_points + pos.view(1, -1, 3)
#         return refined_points

### TEMNPORARY PART ###
'''

if __name__ == '__main__':

    output_dir = BASE_DIR + "/../inference_result/demonet/"
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    os.makedirs(BASE_DIR + "/../logs", exist_ok=True)
    logging.basicConfig(
        filename=BASE_DIR+"/../logs/demo.log", 
        encoding="utf-8", 
        level=logging.DEBUG, 
        filemode="w"
    )
    logger = logging.getLogger(__name__)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")

    # Build MVP dataset
    print("\nBuilding MVP Dataset...")
    data_path = BASE_DIR + "/../mvp/datasets/"
    train_data_path = data_path + "MVP_Train_CP.h5"
    logger.info(f"data directory: {data_path}")
    load_train_dataset = h5py.File(train_data_path, 'r')
    train_dataset = MVPDataset(load_train_dataset, logger=logger, mv=26)
    logger.info(f"lenght of train dataset: {len(train_dataset)}")
    print(train_dataset)
    print("done")

    # Temporary get a single sample 
    pcd_index = random.randint(0, len(train_dataset))
    partial, complete = train_dataset[pcd_index]
    input_img = misc.get_ptcloud_img(partial)
    complete_img = misc.get_ptcloud_img(complete)
    cv2.imwrite(os.path.join(output_dir, 'partial.jpg'), input_img)
    cv2.imwrite(os.path.join(output_dir, 'complete.jpg'), complete_img)
    logger.info(f"shape of a single partial pointcloud: {partial[0].shape}")

    # Define PoinTr instance
    print("\nBuilding PoinTr...")
    init_config = BASE_DIR + "/../cfgs/PCN_models/PoinTr.yaml"
    pointr_ckp = BASE_DIR + "/../ckpts/PCN_Pretrained.pth"
    config = cfg_from_yaml_file(init_config, root=BASE_DIR+"/../")

    # Build PoinTr
    pointr = builder.model_builder(config.model)
    builder.load_model(pointr, pointr_ckp)
    pointr.to(device)
    pointr.eval()
    # print(pointr)

    print("\nPoinTr inference...")
    input_for_pointr = torch.tensor(partial, dtype=torch.float32).unsqueeze(0).to(device)
    ret = pointr(input_for_pointr)
    raw_output = ret[-1] #.permute(1, 2, 0)
    # print(f"coarse point shape: {ret[0].shape}")
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    dense_img = misc.get_ptcloud_img(dense_points)

    print(f"Sample: {pcd_index}")
    print(f"dense points shape: {raw_output.shape}")
    print(f"complete shape: {torch.tensor(complete, dtype=torch.float32).shape}")
    print("done")

    print("Saving output of PoinTr")
    cv2.imwrite(os.path.join(output_dir, 'fine.jpg'), dense_img)
    logger.info(f"images saved at: {output_dir}")
    print("done")
    
    exit()
    ### TEMPORARY PART ###
    # Initialize network, optimizer, and data (replace `pointr_output` and `complete_point_cloud` with your data)
    refinement_net = RefinementNetwork()
    refinement_net = refinement_net.to(device)
    torch.cuda.empty_cache()
    refined_output = refinement_net(input_for_pointr)
    print(f'Shape of refined output: {refined_output.shape}')


    # Saving output
    print("\nSaving output of GPD... ")
    new_dense_points = refined_output.squeeze(0).detach().cpu().numpy()
    new_img = misc.get_ptcloud_img(new_dense_points)
    cv2.imwrite(os.path.join(output_dir, 'new_fine.jpg'), new_img)
    logger.info(f"output destination: {output_dir}")
    print("done!")

    """
    """

    # exit()
    # optimizer = Adam(refinement_net.parameters(), lr=1e-4)

    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
        
    #     # Get the output from PoinTR and refine it
    #     pointr_output = pointr(partial_input)  # Assuming pointr_network and partial_input are defined
    #     refined_output = refinement_net(pointr_output)
        
    #     # Compute Chamfer Distance loss with the target complete point cloud
    #     loss = chamfer_distance(refined_output, complete)
        
    #     # Backpropagation
    #     loss.backward()
    #     optimizer.step()
        
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    ####################

    """
    # Build algebra
    algebra_dim = int(partial.shape[1])
    metric = [1 for i in range(algebra_dim)]
    print("\nGenerating the algebra...")
    algebra = CliffordAlgebra(metric)
    algebra
    print(f"algebra dimention: \t {algebra.dim}")
    print(f"multivectors elements: \t {sum(algebra.subspaces)}")
    print(f"number of subspaces: \t {algebra.n_subspaces}")
    print(f"subspaces grades: \t {algebra.grades.tolist()}")
    print(f"subspaces dimentions: \t {algebra.subspaces.tolist()}")
    print("done")


    # Define custom model
    # model = PointCloudDeformationNet()
    print("\nBuilding the GPD model")
    gpd = GPD(algebra)
    gpd = InvariantCGENN(
        algebra=algebra,
        in_features=raw_output.shape[1], # Note: this number should be fixed to 16384 (if not consider upsampling/ downlasmpling???)
        hidden_features=64,
        out_features=3,
        restore_dim=16384
    )
    gpd = gpd.to(device)
    param_device = next(gpd.parameters()).device
    logger.info(f"model parameters device: {param_device}") 
    print(gpd.name)
    # print(gpd)
    print("done...", end=" ")

    # TODO: load checkpoints for GPD
    print("processing...")

    # Deform the point cloud (inference after training)
    assert param_device == raw_output.device
    # print(raw_output.shape)
    # Convert input to Multivector
    gpd_input = algebra.embed_grade(raw_output, 1)
    deformed_points = gpd(gpd_input)
    print(f"deformed points shape: {deformed_points.shape}")
    # exit()
    

    # Save the deformed point cloud
    print("\nSaving output of GPD... ")
    ga_dense_points = deformed_points.squeeze(0).detach().cpu().numpy()
    ga_img = misc.get_ptcloud_img(ga_dense_points)
    cv2.imwrite(os.path.join(output_dir, 'ga_fine.jpg'), ga_img)
    logger.info(f"output destination: {output_dir}")


    print("done!")
    """
    



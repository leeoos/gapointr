import os 
import sys
import cv2
import h5py
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
##############################################################################
class KNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20, aggr='max'):
        super(KNNConv, self).__init__()
        self.k = k
        self.aggr = aggr
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, pos, features):
        batch_size, num_points, _ = pos.shape
        
        # Reshape pos and features to [N, 3] and [N, in_channels] for distance calculation
        pos = pos.view(-1, 3)
        features = features.view(-1, features.size(-1))

        # Compute pairwise distances
        dists = torch.cdist(pos, pos)  # Shape: [N, N]

        # Find the k-nearest neighbors
        knn_indices = dists.topk(self.k, largest=False).indices  # Shape: [N, k]

        # Gather the neighbor features
        # Expand knn_indices to have an additional dimension for features
        knn_indices_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, features.size(-1))
        print(f'features {features.shape}')
        print(f'indices {knn_indices.shape}')
        # neighbor_features = torch.gather(features, 0, knn_indices)
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
    
class RefinementNetwork(nn.Module):
    def __init__(self, k=20):
        super(RefinementNetwork, self).__init__()
        
        self.conv1 = KNNConv(in_channels=6, out_channels=64, k=k, aggr='max')
        self.conv2 = KNNConv(in_channels=64, out_channels=128, k=k, aggr='max')
        
        # Fully connected layers for refining the point cloud
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
    
    def forward(self, pos):
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
        
        return refined_points + pos

##############################################################################

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
    
    ### TEMPORARY PART ###
    # Initialize network, optimizer, and data (replace `pointr_output` and `complete_point_cloud` with your data)
    refinement_net = RefinementNetwork()
    refinement_net = refinement_net.to(device)
    torch.cuda.empty_cache()
    refined_output = refinement_net(raw_output)
    print(f'Shape of refined output: {refined_output.shape}')


    # Saving output
    print("\nSaving output of GPD... ")
    new_dense_points = refined_output.squeeze(0).detach().cpu().numpy()
    new_img = misc.get_ptcloud_img(new_dense_points)
    cv2.imwrite(os.path.join(output_dir, 'new_fine.jpg'), new_img)
    logger.info(f"output destination: {output_dir}")
    print("done!")
    exit()

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
    



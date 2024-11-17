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
from torch_geometric.nn import knn_graph

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# PoinTr imports
from utils import misc
from tools import builder
from utils.config import cfg_from_yaml_file

# Dataset  
from mvp.mvp_dataset import MVPDataset

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra

# Models
from models.ga_models.GAFold import GAFold

# Metrics
from extensions.chamfer_dist import (
    ChamferDistanceL1, 
    ChamferDistanceL2
)

if __name__ == '__main__':

    output_dir = BASE_DIR + "/../results/demonet/"
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

    # Build MVP test dataset
    print("\nBuilding MVP Dataset...")
    data_path = BASE_DIR + "/../mvp/datasets/"
    train_data_path = data_path + "MVP_Test_CP.h5"
    logger.info(f"data directory: {data_path}")
    load_train_dataset = h5py.File(train_data_path, 'r')
    train_dataset = MVPDataset(load_train_dataset, logger=logger, mv=26)
    logger.info(f"lenght of train dataset: {len(train_dataset)}")
    print("done")

    # Temporary get a single sample 
    random.seed(None) # reset seed to get random sample
    pcd_index = random.randint(0, len(train_dataset))
    # print(pcd_index)
    partial, complete = train_dataset[pcd_index]
    input_img = misc.get_ptcloud_img(partial)
    complete_img = misc.get_ptcloud_img(complete)
    cv2.imwrite(os.path.join(output_dir, 'partial.jpg'), input_img)
    cv2.imwrite(os.path.join(output_dir, 'complete.jpg'), complete_img)
    logger.info(f"shape of a single partial pointcloud: {partial[0].shape}")

    # Build algebra
    algebra_dim = int(partial.shape[1])
    metric = [1 for i in range(algebra_dim)]
    print("\nBuilding the algebra...")
    algebra = CliffordAlgebra(metric)
    print(f"algebra dimention: \t {algebra.dim}")
    print(f"multivectors elements: \t {sum(algebra.subspaces)}")
    print(f"number of subspaces: \t {algebra.n_subspaces}")
    print(f"subspaces grades: \t {algebra.grades.tolist()}")
    print(f"subspaces dimentions: \t {algebra.subspaces.tolist()}")
    print("done")

    # Define PoinTr instance
    print("\nBuilding PoinTr...")
    # config_type = "KITTI_models"
    config_type = "PCN_models"
    # config_type = "ShapeNet34_models"
    # config_type = "ShapeNet55_models"
    init_config = BASE_DIR + "/../cfgs/" + config_type + "/PoinTr.yaml"
    pointr_ckpt = BASE_DIR + "/../ckpts/" + config_type +"/pointr.pth"
    config = cfg_from_yaml_file(init_config, root=BASE_DIR+"/../")

    # Build PoinTr
    pointr = builder.model_builder(config.model)
    builder.load_model(pointr, pointr_ckpt)
    pointr.to(device)
    pointr.eval()

    # PoinTr parametr estimation
    total_params = sum(p.numel() for p in pointr.parameters())
    param_size_bytes = total_params * 4  # Assuming float32
    model_size_mb = param_size_bytes / (1024 ** 2)
    print(f"Total Parameters: {total_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")

    print("\nPoinTr inference...")
    input_for_pointr = torch.tensor(partial, dtype=torch.float32).unsqueeze(0).to(device)
    ret = pointr(input_for_pointr)
    raw_output = ret[1] #.permute(1, 2, 0)
    pointr_parametrs = ret[-1]
    dense_points = raw_output.squeeze(0).detach().cpu().numpy()
    dense_img = misc.get_ptcloud_img(dense_points)

    print(f"input sample shape: {input_for_pointr.shape}")
    print(f"coarse points shape: {ret[0].shape}")
    print(f"dense points shape: {raw_output.shape}")
    print(f"complete reference shape: {torch.tensor(complete, dtype=torch.float32).shape}")
    print("done")

    print("\nSaving output of PoinTr")
    cv2.imwrite(os.path.join(output_dir, 'fine.jpg'), dense_img)
    logger.info(f"images saved at: {output_dir}")
    print("done")

    ### TEMPORARY PART ###
    print("\nBuilding GAPoinTr...")
    ga_checkpoints = os.path.join(
        BASE_DIR, 
        # f"../saves/training/{config_type.lower()}_train_0/model_state_dict.pt"
        f"../saves/training/pcn_models_1/model_state_dict.pt"
    )
    print(f"Loading checkpoints from: {ga_checkpoints}")
    model = GAFold(
        algebra=algebra,  
        embed_dim=8
    )
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            ga_checkpoints,
            weights_only=True
        )
    )
    model.eval()
    torch.cuda.empty_cache()

    # GAPoinTr parametr estimation
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = total_params * 4  # Assuming float32
    model_size_mb = param_size_bytes / (1024 ** 2)
    print(f"Total Parameters: {total_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")

    print("\nGAPoinTr inference...")
    output = model(input_for_pointr, pointr, pointr_parametrs)
    print(f'Shape of refined output: {output.shape}')
    print("done")

    # Saving output
    print("\nSaving output of GAPoinTr... ")
    new_dense_points = output.squeeze(0).detach().cpu().numpy()
    new_img = misc.get_ptcloud_img(new_dense_points)
    cv2.imwrite(os.path.join(output_dir, 'new_fine.jpg'), new_img)
    logger.info(f"output destination: {output_dir}")
    print("done")

    # Chamfer Distance helper function
    print("\nQuantitative evaluation")
    chamfer_dist_l1 = ChamferDistanceL1()
    complete_tensor = torch.tensor(complete, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Chamfer distance Partial: {chamfer_dist_l1(ret[0], complete_tensor)}")
    print(f"Chamfer distance PoinTr: {chamfer_dist_l1(raw_output, complete_tensor)}")
    print(f"Chamfer distance GAPoinTr: {chamfer_dist_l1(output, complete_tensor)}")

    # Pointr
    x_max = torch.max(raw_output[:,:,0]).item()
    y_max = torch.max(raw_output[:,:,1]).item()
    z_max = torch.max(raw_output[:,:,2]).item()
    x_min = torch.min(raw_output[:,:,0]).item()
    y_min = torch.min(raw_output[:,:,1]).item()
    z_min = torch.min(raw_output[:,:,2]).item()
    print(f"Max for PoinTr: {x_max, y_max, z_max}")
    print(f"Min for PoinTr: {x_min, y_min, z_min}")

    # GAPoinTr
    x_max = torch.max(output[:,:,0]).item()
    y_max = torch.max(output[:,:,1]).item()
    z_max = torch.max(output[:,:,2]).item()
    x_min = torch.min(output[:,:,0]).item()
    y_min = torch.min(output[:,:,1]).item()
    z_min = torch.min(output[:,:,2]).item()
    print(f"Max for GAPoinTr: {x_max, y_max, z_max}")
    print(f"Min for GAPoinTr: {x_min, y_min, z_min}")
    print("All done!")

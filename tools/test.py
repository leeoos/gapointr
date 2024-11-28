import os 
import gc
import sys
import cv2
import json
import h5py
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
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
from torch.utils.data import DataLoader

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra

# Models
from models.GAPoinTr import GAFeatures

# Metrics
from extensions.chamfer_dist import (
    ChamferDistanceL1, 
    ChamferDistanceL2
)

if __name__ == '__main__':

    run_name = "cdl1_sparse_dense_1"

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build MVP test dataset
    print("\nBuilding MVP Dataset...")
    data_path = BASE_DIR + "/../mvp/datasets/"
    test_data_path = data_path + "MVP_Test_CP.h5"
    load_test_dataset = h5py.File(test_data_path, 'r')
    test_dataset = MVPDataset(load_test_dataset, logger=None)

    # Make dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    print("done")

    # Build algebra
    algebra_dim = int(3)
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

    
    print("\nBuilding GAPoinTr...")
    ga_checkpoints = os.path.join(
        BASE_DIR, 
        f"../saves/training/{config_type}/{run_name}/model_state_dict.pt"
        # f"../saves/training//model_state_dict.pt"
    )
    print(f"Loading checkpoints from: {ga_checkpoints}")
    model = GAFeatures(
        algebra=algebra,  
        embed_dim=8,
        hidden_dim=256,
        pointr=pointr
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

    print("\n Testing")
    chdl1_fn = ChamferDistanceL1()
    chdl2_fn = ChamferDistanceL2()
    collection = {
        'GAPoinTr': {
              'chdl1': 0,
              'chdl2': 0,
              'dense_coarse': 0,
        },
        'PoinTr': {
              'chdl1': 0,
              'chdl2': 0,
              'dense_coarse': 0,
        },
    }
    flush_step = 500
    with tqdm(total=len(test_dataloader)) as bbar:
        for step, pcd in enumerate(test_dataloader):

            partial, complete = pcd

            # Send point clouds to device
            partial = partial.to(device)
            complete = complete.to(torch.float32).to(device)

            # Pass partial pcd to PoinTr
            pointr_parameters = {}
            with torch.no_grad():
                pointr_output = pointr(partial)
                pointr_parameters = pointr_output[-1]
                output = model(pointr, pointr_parameters)


            # PoinTr
            collection["PoinTr"]['chdl1'] += chdl1_fn(pointr_output[1], complete).item()
            collection["PoinTr"]['chdl1'] += chdl2_fn(pointr_output[1], complete).item()
            collection["PoinTr"]['dense_coarse'] += chdl1_fn(pointr_output[1], complete).item() + chdl1_fn(pointr_output[0], complete).item()

            # Model
            collection["GAPoinTr"]['chdl1'] += chdl1_fn(output, complete).item()
            collection["GAPoinTr"]['chdl2'] += chdl2_fn(output, complete).item()
            model_dense_coarse = chdl1_fn(output, complete) + chdl1_fn(pointr_output[0], complete)
            collection["GAPoinTr"]['dense_coarse'] += model_dense_coarse.item()

            collection["GAPoinTr"]['chdl1']
            

            bbar.set_postfix(batch_loss=model_dense_coarse.item())
            bbar.update(1)

            # free up cuda mem
            del complete
            del partial
            del output
            if step > 0 and step % flush_step == 0:
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

    collection["PoinTr"]['chdl1'] /= (step + 1)
    collection["PoinTr"]['chdl2'] /= (step + 1)
    collection["PoinTr"]['dense_coarse'] /= (step + 1)
    collection["GAPoinTr"]['chdl1'] /= (step + 1)
    collection["GAPoinTr"]['chdl2'] /= (step + 1)
    collection["GAPoinTr"]['dense_coarse'] /= (step + 1)
                
    print(collection)

    out_dir = os.path.join(BASE_DIR, "..", "results/metrics")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + "/collection.json", "w") as jout:
        json.dump(collection, jout, indent=4)

    print("All done!")

import os 
import sys
import cv2
import h5py
import torch
import logging
import numpy as np
import torch.nn as nn

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
    pcd_index = 20
    partial, complete = train_dataset[pcd_index]
    input_img = misc.get_ptcloud_img(partial)
    complete_img = misc.get_ptcloud_img(complete)
    cv2.imwrite(os.path.join(output_dir, 'partial.jpg'), input_img)
    cv2.imwrite(os.path.join(output_dir, 'complete.jpg'), complete_img)
    logger.info(f"shape of a single partial pointcloud: {partial[0].shape}")

    # exit()

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
    print(pointr)

    print("\nPoinTr inference...")
    input_for_pointr = torch.tensor(partial, dtype=torch.float32).unsqueeze(0).to(device)
    ret = pointr(input_for_pointr)
    raw_output = ret[-1] #.permute(1, 2, 0)
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    dense_img = misc.get_ptcloud_img(dense_points)
    print("done")

    print("Saving output of PoinTr")

    cv2.imwrite(os.path.join(output_dir, 'fine.jpg'), dense_img)
    logger.info(f"images saved at: {output_dir}")
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
    print(gpd)
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
    



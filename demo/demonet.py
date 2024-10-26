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
from ga_fold import PointCloudGADeformationNet
from mvp.mvp_dataset import MVPDataset


if __name__ == '__main__':

    # Setup logging
    # Logger
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

    # Load the original point cloud
    # input_pcd = "./airplane.pcd"
    # points = load_point_cloud(input_pcd)
    # print(f"Point types: {type(points)}")
    # print(f"Points shape: {points.shape}")

    # Build MVP dataset
    print("\nBuilding MVP Dataset...")
    data_path = BASE_DIR + "/../mvp/datasets/"
    train_data_path = data_path + "MVP_Train_CP.h5"
    logger.info(f"data directory: {data_path}")
    load_train_dataset = h5py.File(train_data_path, 'r')
    train_dataset = MVPDataset(load_train_dataset, logger=logger)
    logger.info(f"lenght of train dataset: {len(train_dataset)}")
    print(train_dataset)
    print("done")

    # Temporary get a single sample 
    points = train_dataset[420][0] # get partial pcd
    logger.info(f"shape of a single partial pointcloud: {points[0].shape}")

    # Build algebra
    algebra_dim = int(points.shape[1])
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
    input_for_pointr = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)
    ret = pointr(input_for_pointr)
    raw_output = ret[-1].permute(1, 2, 0)
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    input_img = misc.get_ptcloud_img(points)
    dense_img = misc.get_ptcloud_img(dense_points)
    print("done")

    print("Saving output of PoinTr")
    cv2.imwrite(os.path.join(BASE_DIR, 'input.jpg'), input_img)
    cv2.imwrite(os.path.join(BASE_DIR, 'fine.jpg'), dense_img)
    logger.info(f"images saved at: {BASE_DIR}")
    print("done")


    # Define custom model
    # model = PointCloudDeformationNet()
    print("\nBuilding the GA deformer model")
    ga_model = PointCloudGADeformationNet(algebra)
    ga_model = ga_model.to(device)
    param_device = next(ga_model.parameters()).device
    logger.info(f"model parameters device: {param_device}") 
    print(ga_model)
    print("done...", end=" ")
    print("processing...")

    # Deform the point cloud (inference after training)
    assert param_device == raw_output.device
    print(raw_output.shape)
    deformed_points = ga_model(raw_output)
    print(type(deformed_points))
    print(deformed_points.shape)
    

    # Save the deformed point cloud
    print("Saving output of GA deformer... ")
    ga_dense_points = deformed_points.detach().cpu().numpy()
    ga_img = misc.get_ptcloud_img(ga_dense_points)
    cv2.imwrite(os.path.join(BASE_DIR, 'ga_fine.jpg'), ga_img)
    logger.info(f"output destination: {BASE_DIR}")

    
    # output_pcd = "deformed_output.pcd"
    # logger.info(f"saving destination: {output_pcd}")
    # save_deformed_point_cloud(deformed_points.cpu().detach().numpy(), output_pcd)
    # points = load_point_cloud(output_pcd)
    # print("done")

    # # Convert point cloud to image
    # print("Generating final image")
    # point_cloud_to_image_with_color(points, img_size=(256, 256), output_file="ga_fine.png")

    print("done!")
    



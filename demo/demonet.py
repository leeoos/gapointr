import math
import os
import pprint 
import sys
import cv2
import h5py
import yaml
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from easydict import EasyDict

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# PoinTr imports
from utils import misc
from tools import builder
from utils.config import cfg_from_yaml_file, model_info

# Dataset  
from mvp.mvp_dataset import MVPDataset

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra

# Models
from models.PoinTr import PoinTr
from models.PoinTrWrapper import PoinTrWrapper

# Loss
from pga_lib.pgaloss import PGALoss


# Metrics
from pointnet2_pytorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import (
    ChamferDistanceL1, 
    ChamferDistanceL2
)
from utils.metrics import Metrics
from utils.dist_utils import fps


def demo(config, idx=None):
    
    # Get latest version
    version = config['run_name'] 
    ft_version = config['ft_version'] 
    step = config['step']
    train_type = "experiments" if config['pretrained'] else "full"
    save_dir = os.path.join(
        BASE_DIR,
        '..',
        config['save_path'], 
        config['pointr_config'],
        train_type,
    )
    # run_counter = -1
    # for file in os.listdir(save_dir):
    #     if version.split('_') == file.split('_')[:-1]: 
    #         run_counter += 1
    # version = version + "_" + str(run_counter)
    print(f"Demo verison: {version}")

    # Set up output
    output_dir = BASE_DIR + f"/../results/experiments/demonet/{version}"
    os.makedirs(output_dir, exist_ok=True)
    config_type = config['pointr_config'] 

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
    train_dataset = MVPDataset(load_train_dataset, transform_for='train', logger=logger, mv=26)
    logger.info(f"lenght of train dataset: {len(train_dataset)}")
    print("done")

    # Get a single sample from test set for demonstartion
    random.seed(None) # reset seed to get random sample
    if not idx:
        pcd_index = random.randint(1, len(train_dataset))
    else:
        pcd_index = idx
    sample_class = train_dataset.__get_class__(pcd_index)
    print(f"Selected sample: {pcd_index}")
    partial, complete = train_dataset[pcd_index]

    # This is needed in case preprocess is applied
    if isinstance(partial, torch.Tensor):
        print("Preprocessing has been applied on data samples!")
        print(f"Sample tensor of shape: {partial.shape}")
        partial_pc = partial.detach().numpy()
        complete_pc = complete.detach().numpy()
        input_for_pointr = partial.clone().unsqueeze(0).to(device)
        complete_shape = complete.shape
        complete_tensor = complete.clone().unsqueeze(0).to(device)
    else:
        partial_pc = partial
        complete_pc = complete
        input_for_pointr = torch.tensor(partial, dtype=torch.float32).unsqueeze(0).to(device)
        complete_shape = torch.tensor(complete, dtype=torch.float32).shape
        complete_tensor = torch.tensor(complete, dtype=torch.float32).unsqueeze(0).to(device)

    # Get input images form dataet samples
    input_img = misc.get_ptcloud_img(partial_pc)
    complete_img = misc.get_ptcloud_img(complete_pc)
    cv2.imwrite(os.path.join(output_dir, 'partial.jpg'), input_img)
    cv2.imwrite(os.path.join(output_dir, 'complete.jpg'), complete_img)
    logger.info(f"shape of a single partial pointcloud: {partial[0].shape}")


    # TEST LOSS FUNCTION
    if config['debug']:
        # Define PoinTr instance
        print("\nBuilding PoinTr...")
        init_config = BASE_DIR + "/../cfgs/" + config_type + "/PoinTr.yaml"
        pointr_ckpt = BASE_DIR + "/../ckpts/" + config_type +"/pointr.pth"
        pointr_config = cfg_from_yaml_file(init_config, root=BASE_DIR+"/../")
        pointr = builder.model_builder(pointr_config.model)
        builder.load_model(pointr, pointr_ckpt)
        pointr.to(device)
        pointr.eval()
        model_info(pointr)

        # PoinTr inference
        # print("\nPoinTr inference...")
        ret = pointr(input_for_pointr)
        pointr_output = ret[1] #.permute(1, 2, 0)
        pointr_coarse = ret[0]
        dense_points = pointr_output.squeeze(0).detach().cpu().numpy()
        coarse_points = ret[0].squeeze(0).detach().cpu().numpy()
        dense_img = misc.get_ptcloud_img(dense_points)
        coarse_img = misc.get_ptcloud_img(coarse_points)
        # print(f"input sample shape: {input_for_pointr.shape}")
        # print(f"complete reference shape: {complete_shape}")
        # print(f"coarse points shape: {ret[0].shape}")
        # print(f"dense points shape: {pointr_output.shape}")
        # print("done")

        print("\nSaving output of PoinTr...")
        cv2.imwrite(os.path.join(output_dir, 'fine.jpg'), dense_img)
        cv2.imwrite(os.path.join(output_dir, 'coarse.jpg'), coarse_img)
        logger.info(f"images saved at: {output_dir}")
        print("done")

        loss_fn = PGALoss()
        fake_source_points_1 = torch.randn((1, 2048, 3), requires_grad=True).to('cuda')  # Input partial point cloud
        fake_source_points_2 = torch.randn((1, 2048, 3), requires_grad=True).to('cuda')  # Input partial point cloud
        real_source = input_for_pointr.detach().clone().requires_grad_(True)
        target = complete_tensor.detach().clone().requires_grad_(True)
        print("\nHigh")
        high_loss = loss_fn((fake_source_points_1, fake_source_points_2) , target)
        print(f"high loss: {high_loss.item()}\n")
        print("Medium")
        medium_loss = loss_fn((input_for_pointr, input_for_pointr) , target)
        print(f"medium loss: {medium_loss.item()}\n")
        print("Low")
        low_loss = loss_fn((target, target), target)
        print(f"low loss: {low_loss.item()}\n")
        print("Pointr")
        pointr_loss = loss_fn(ret, target)
        print(f"pointr loss: {pointr_loss.item()}\n")

        print("Attempting backward pass!")
        print(target.requires_grad)
        print(ret[0].requires_grad)
        print(pointr_loss.requires_grad)
        print(pointr_loss.grad_fn)
        pointr_loss.backward()
        print("done!")
        exit()
    ####################

    # PoinTr
    if ft_version:
        print(f"\nFine-tuned PoinTr: {ft_version}")
        ft_checkpoint_file = os.path.join(
            BASE_DIR, 
            f"../saves/{config_type}/{train_type}/{ft_version}/training/{step}/checkpoint.pt"
        )
        print(f"Loading checkpoints from: {ft_checkpoint_file}")
        pointr = PoinTrWrapper(pointr)
        ft_checkpoint = torch.load(ft_checkpoint_file, weights_only=True)
        pointr.load_state_dict(ft_checkpoint['model'], strict=False) # RUN YOU CLEVER BOY AND REMEMBER
        pointr = pointr.to(device)
        model_info(pointr)
        pt_output = pointr(input_for_pointr)[-1]
    else:
        print("\nBuilding PoinTr...")
        init_config = BASE_DIR + "/../cfgs/" + config_type + "/PoinTr.yaml"
        pointr_ckpt = BASE_DIR + "/../ckpts/" + config_type +"/pointr.pth"
        pt_config = cfg_from_yaml_file(init_config, root=BASE_DIR+"/../")
        pointr = builder.model_builder(pt_config.model)
        builder.load_model(pointr, pointr_ckpt)
        pointr.to(device)
        pointr.eval()
        model_info(pointr)
        pt_output = pointr(input_for_pointr)[-1]

    # Saving output
    print(f"\nSaving output of PoinTr... {ft_version}")
    pt_dense_points = pt_output.squeeze(0).detach().cpu().numpy()
    new_img = misc.get_ptcloud_img(pt_dense_points)
    cv2.imwrite(os.path.join(output_dir, 'pointr.jpg'), new_img)
    logger.info(f"output destination: {output_dir}")
    print("done")

    # Custom Model
    print(f"\nBuilding Custom model: {config['ga_version']}")
    # checkpoint_file = os.path.join(
    #     BASE_DIR, 
    #     f"../saves/{config_type}/{train_type}/{version}/training/{step}/checkpoint.pt"
    # )
    checkpoint_file = config['ga_version']
    print(f"Loading checkpoints from: {checkpoint_file}")
    model = PoinTrWrapper(pointr, gafet=config['gafet'])
    checkpoint = torch.load(checkpoint_file, weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=False) # RUN YOU CLEVER BOY AND REMEMBER
    model = model.to(device)
    model_info(model)
    output = model(input_for_pointr)[-1]

    # Saving output
    print("\nSaving output of GAPoinTr... ")
    new_dense_points = output.squeeze(0).detach().cpu().numpy()
    new_img = misc.get_ptcloud_img(new_dense_points)
    cv2.imwrite(os.path.join(output_dir, 'ga.jpg'), new_img)
    logger.info(f"output destination: {output_dir}")
    print("done")

    # Compute the metrics
    # print("\nPoinTr Evaluation:")
    # to_test = fps(pointr_output.detach(), num=2048)
    # print(to_test.shape)
    # metrics = Metrics.get(to_test, complete_tensor, require_emd=True)
    # metric_names = Metrics.names()
    # for name, value in zip(metric_names, metrics):
    #     print(f"{name}: {value.item()}")

    print(f"\nPoinTr {ft_version} Evaluation:")
    to_test = fps(pt_output.detach(), num=2048)
    print(to_test.shape)
    metrics = Metrics.get(to_test, complete_tensor, require_emd=True)
    metric_names = Metrics.names()
    pointr_metrics = []
    for name, value in zip(metric_names, metrics):
        print(f"{name}: {value.item()}")
        pointr_metrics.append(value.item())

    print("\nGAPoinTr Evaluation:")
    to_test = fps(output.detach(), num=2048)
    metrics = Metrics.get(to_test.detach(), complete_tensor, require_emd=True)
    metric_names = Metrics.names()
    ga_pointr_metrics = []
    for name, value in zip(metric_names, metrics):
        print(f"{name}: {value.item()}")
        ga_pointr_metrics.append(value.item())
        
        
    if config['cherry_pick']:
        if ga_pointr_metrics[0] - 5  >  pointr_metrics[0]: return False
        if ga_pointr_metrics[1] + 5 <  pointr_metrics[1]: return False
        if ga_pointr_metrics[2] + 5 <  pointr_metrics[2]: return False
        if ga_pointr_metrics[3] + 5 <  pointr_metrics[3]: return False

    print("All done!")

    return pcd_index, sample_class



if __name__ == '__main__':

    # Get info from config
    training_config = os.path.join(BASE_DIR, '../cfgs/GAPoinTr-training.yaml')
    with open(training_config, "r") as file:
        config = yaml.safe_load(file)

    if config['cherry_pick']:

        best_values = {}
        original_version = config['gafet']
        # for version in ['head', 'fold', 'backbone']:
        for version in ['fold']:
            config['gafet'] = config['gafet'].replace(original_version, version)
            config['run_name'] = config['run_name'].replace(original_version, version)
            config['ga_version'] = config['ga_version'].replace(original_version, version)
            original_version = config['gafet']

            sample = demo(config)
            while not sample:
                sample = demo(config)
                
            best_values[version] = sample
            # exit()
        
        print(best_values)

    else:

        # step: 32642
        # puff: 7190
        # stairs: 33159
        # box: 35136
        # barbequeue: 41348

        # cassettiera: 34394 
        # aereo: 26383 
        # moto: 38999 
        # letto: 32449
        # plane: 23482
        # table  4314
        # another table: 6487
        # nice table: 4101

        # 11854

        idx = 7191
        # demo(config, idx)
        # ref = math.floor(idx / 26)
        original_version = config['gafet']
        for version in ['head', 'fold', 'backbone']:
        # for version in ['head']:
        # for version in ['backbone']:
        # for version in ['fold']:
            config['gafet'] = config['gafet'].replace(original_version, version)
            config['run_name'] = config['run_name'].replace(original_version, version)
            config['ga_version'] = config['ga_version'].replace(original_version, version)
            original_version = config['gafet']
            idx, _ = demo(config, idx)






    # # Chamfer Distance helper function
    # print("\nQuantitative evaluation")
    # chamfer_dist_l1 = ChamferDistanceL1()
    # print(f"Chamfer distance Partial: {chamfer_dist_l1(input_for_pointr, complete_tensor)}")
    # print(f"Chamfer distance Coarse: {chamfer_dist_l1(pointr_coarse, complete_tensor)}")
    # print(f"Chamfer distance Fine: {chamfer_dist_l1(pointr_output, complete_tensor)}")
    # print(f"Chamfer distance GAPoinTr: {chamfer_dist_l1(output, complete_tensor)}\n")

    # # Pointr
    # x_max = torch.max(pointr_output[:,:,0]).item()
    # y_max = torch.max(pointr_output[:,:,1]).item()
    # z_max = torch.max(pointr_output[:,:,2]).item()
    # x_min = torch.min(pointr_output[:,:,0]).item()
    # y_min = torch.min(pointr_output[:,:,1]).item()
    # z_min = torch.min(pointr_output[:,:,2]).item()
    # print(f"Max for PoinTr: {x_max, y_max, z_max}")
    # print(f"Min for PoinTr: {x_min, y_min, z_min}")

    # # GAPoinTr
    # x_max = torch.max(output[:,:,0]).item()
    # y_max = torch.max(output[:,:,1]).item()
    # z_max = torch.max(output[:,:,2]).item()
    # x_min = torch.min(output[:,:,0]).item()
    # y_min = torch.min(output[:,:,1]).item()
    # z_min = torch.min(output[:,:,2]).item()
    # print(f"Max for GAPoinTr: {x_max, y_max, z_max}")
    # print(f"Min for GAPoinTr: {x_min, y_min, z_min}")

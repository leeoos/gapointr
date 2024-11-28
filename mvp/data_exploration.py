import os
import h5py
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# This should go in the main file 
# file_path = os.path.realpath(__file__)
file_path = os.path.abspath(os.path.dirname(__file__))
data_path = file_path + "/datasets/"
train_data_path = data_path + "MVP_Train_CP.h5"
print(f"Data directory: {data_path}")
train_dataset = h5py.File(train_data_path, 'r') 

# This should be a function
# Load H5 data
print(f"Dataset: {train_dataset}")
print(f"INFO:")
print(f"datasets keys: {list(train_dataset.keys())}")
# output: ['complete_pcds', 'incomplete_pcds', 'labels']
print(f"dataset format: {type(train_dataset['complete_pcds'])}")
print(f"data sample format: {type(train_dataset['complete_pcds'][0])}")
print(f"complete point clouds : {train_dataset['complete_pcds']}")
print(f"incomplete point clouds : {train_dataset['incomplete_pcds']}")
print(f"first sample: {train_dataset['complete_pcds'][0]}")
print(f"shape: {train_dataset['complete_pcds'][0].shape}")

# This should be another function
# Data visualization
# plt.ion() # enable interactive mode
fig = plt.figure()
ax = plt.axes(projection='3d')
random_sample = np.random.randint(0, len(train_dataset['complete_pcds']))

# Complete sample
first_x = train_dataset['complete_pcds'][random_sample][:,0]
first_y = train_dataset['complete_pcds'][random_sample][:,1]
first_z = train_dataset['complete_pcds'][random_sample][:,2]
ax.scatter3D(first_x, first_y, first_z, color='blue', label=f'Complete sample {random_sample}')

# Complete sample
first_x = train_dataset['incomplete_pcds'][random_sample][:,0]
first_y = train_dataset['incomplete_pcds'][random_sample][:,1]
first_z = train_dataset['incomplete_pcds'][random_sample][:,2]
ax.scatter3D(first_x, first_y, first_z, color='red', label=f'Incomplete sample {random_sample}')

# Show 
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()
plt.pause(0.001)
plt.show()


"""
import os
import gc
import sys
import cv2
import json
import h5py
import yaml
import torch
import logging
# from torchinfo import summary # pip install torchinfo
from torch.utils.data import DataLoader
from torch.optim import (
    AdamW
)
from torch.optim.lr_scheduler import LambdaLR
availabel_optimizers = {
    'AdamW': AdamW,
    'LambdaLR': LambdaLR,
}

import random

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# PoinTr imports
from utils import misc
from tools import builder
from tools import training
from utils.config import (
    cfg_from_yaml_file,  
    get_instance, 
    model_info,
    model_summary
)
from extensions.chamfer_dist import (
    ChamferDistanceL1, 
    ChamferDistanceL2
)
from pointnet2_ops import pointnet2_utils

# MVP Dataset  
from mvp.mvp_dataset import MVPDataset

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra

# Models
# from models.GAPoinTr import GAFeatures

def main():
    # Here insert argparser and select between train and test

    # Load training configuration file
    training_config = os.path.join(BASE_DIR, 'cfgs/GAPoinTr-training.yaml')
    with open(training_config, "r") as file:
        config = yaml.safe_load(file)
    
    # Setup logging
    os.makedirs(os.path.join(BASE_DIR , "logs"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(BASE_DIR, "logs/train.log"), 
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
    logger.info(f"Datasets: {config['train_dataset']}\t{config['test_dataset']}")
    train_data_path = config['train_dataset'] #data_path + "MVP_Train_CP.h5"
    load_train_dataset = h5py.File(train_data_path, 'r')
    train_dataset = MVPDataset(load_train_dataset, logger=logger)
    logger.info(f"lenght of train dataset: {len(train_dataset)}")
    print((f"Lenght of train dataset: {len(train_dataset)}"))
    test_data_path = config['test_dataset'] #data_path + "MVP_Test_CP.h5"
    load_test_dataset = h5py.File(test_data_path, 'r')
    test_dataset = MVPDataset(load_test_dataset, logger=logger)
    logger.info(f"lenght of test dataset: {len(test_dataset)}")

    # Make dataloader
    # config['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])
    logger.info(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print("done")

    random.seed(None)
    rand_stop = random.randint(0, len(train_dataloader))
    # rand_stop = 0 
    for step, batch  in enumerate(train_dataloader):
        partial, complete = batch #train_dataset[rand_stop] #batch
        partial = partial.squeeze().numpy()
        complete = complete.squeeze().numpy()
        # print(partial.shape)
        # print(type(partial))
        # exit()
        if step == rand_stop:
            print(step)
            input_img = misc.get_ptcloud_img(partial)
            cv2.imwrite(os.path.join("./", 'partial.jpg'), input_img)
            complete_img = misc.get_ptcloud_img(complete)
            cv2.imwrite(os.path.join("./", 'complete.jpg'), complete_img)
            break
    exit()

"""
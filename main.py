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

# Dataset
from datasets import build_dataset_from_cfg  
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
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])
    logger.info(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print("done")

    # Build algebra
    # points = train_dataset[42][0] # get partial pcd to build the algebra
    # logger.info(f"shape of a single partial pointcloud: {points[0].shape}")
    # algebra_dim = int(points.shape[1])
    # metric = [1 for i in range(algebra_dim)]
    # print("\nGenerating the algebra...")
    # algebra = CliffordAlgebra(metric)
    # print(f"algebra dimention: \t {algebra.dim}")
    # print(f"multivectors elements: \t {sum(algebra.subspaces)}")
    # print(f"number of subspaces: \t {algebra.n_subspaces}")
    # print(f"subspaces grades: \t {algebra.grades.tolist()}")
    # print(f"subspaces dimentions: \t {algebra.subspaces.tolist()}")
    # print("done")

    # Define PoinTr instance
    print("\nBuilding PoinTr...")
    pointr_init_config = os.path.join(
        BASE_DIR, "cfgs", config['pointr_config'], "PoinTr.yaml"
    )
    pointr_ckp = os.path.join(BASE_DIR, "ckpts", config['pointr_config'], "pointr.pth")
    pointr_config = cfg_from_yaml_file(pointr_init_config, root=BASE_DIR+"/")
    pointr = builder.model_builder(pointr_config.model)
    builder.load_model(pointr, pointr_ckp)
    pointr = pointr.to(device)
    model_info(pointr)

    # print("\nBuilding GAPoinTr...")
    # gapointr_init_config = os.path.join(
    #     BASE_DIR, "cfgs", config['pointr_config'], "GAPoinTr.yaml"
    # )
    # with open(gapointr_init_config, "r") as ga_file:
    #     ga_config = yaml.safe_load(ga_file)
    # gapointr_ckp = os.path.join(BASE_DIR, "ckpts", config['pointr_config'], "pointr.pth")
    # gapointr_config = cfg_from_yaml_file(gapointr_init_config, root=BASE_DIR+"/")
    # gapointr = builder.model_builder(gapointr_config.model)
    # builder.load_model(gapointr, gapointr_ckp)
    # gapointr = gapointr.to(device)
    # model_info(gapointr)

    # # Torch Info: pip install torchinfo
    # if config['debug']: 
    #     model_summary(gapointr)
    #     input_shape = train_dataset[42][0].shape
    #     summary(gapointr, input_size=(config['batch_size'], *input_shape))


    # Define custom model
    # print("\nBuilding GAPoinTr...")
    # gapointr = GAFeatures(
    #     algebra=algebra,  
    #     embed_dim=config['embed_dim'],
    #     hidden_dim=256,
    #     pointr=pointr
    # )
    # gapointr = gapointr.to(device)
    # param_device = next(gapointr.parameters()).device
    # logger.info(f"model parameters device: {param_device}") 
    # model_info(gapointr)

    # Build parameters from yaml
    # model_parameters = list(pointr.parameters()) + list(gapointr.parameters())
    # model_parameters = list(pointr.parameters())
    # if ga_config['model']['ga_head']:
    #     model_parameters = list(gapointr.ga_transformer.parameters()) + list(gapointr.project_back.parameters()) + list(gapointr.foldingnet.parameters())
    # elif ga_config['model']['ga_tail']:
    #     model_parameters = list(gapointr.base_model.grouper.input_trans.parameters()) 
    # else:
    #     model_parameters = list(gapointr.parameters())

    # # if config['debug']: print(model_parameters)
    model_parameters = pointr.parameters()
    trainer_parameters = {
        'epochs': config['epochs'],
        'optimizer': get_instance(
                        config['optimizer'], 
                        availabel_optimizers,
                        {"params": model_parameters}
                    ),
        'scheduler': None,
        'device': device,
        'losses':  {
            'ChDL1': ChamferDistanceL1(),
            'ChDL2': ChamferDistanceL2()
        }
    }

    # Set up saving directory
    run_name = 'pointr'
    # if ga_config['model']['ga_head'] and not ga_config['model']['ga_tail']:
    #     run_name += "_head"
    # elif ga_config['model']['ga_tail'] and not ga_config['model']['ga_head']:
    #     run_name += "_tail"
    # elif ga_config['model']['ga_head'] and not ga_config['model']['ga_tail']:
    #     run_name += "_head_tail"

    save_dir = os.path.join(
        BASE_DIR,
        config['save_path'], 
        config['pointr_config'],
    )
    run_counter = 0
    for file in os.listdir(save_dir):
        if run_name.split('_') == file.split('_')[:-1]: 
            run_counter += 1
    if config['override_cache'] and run_counter > 0: run_counter -= 1
    save_dir = os.path.join(save_dir, run_name)
    save_dir = save_dir+"_"+str(run_counter)
    # os.makedirs(save_dir, exist_ok=True)
    logger.info(f"\nSaving checkpoints in: {save_dir}")
    logger.info(f"\nSaving losses in: {save_dir}")

    # Training
    print("\nTraining")
    trainer = training.Trainer(
        parameters=trainer_parameters,
        logger=logger,
        debug=config['debug']
    )
    trainer.train(
        backbone=pointr,
        model=pointr,
        dataloader=train_dataloader,
        save_path=save_dir
    )
    print("\nEnd of training!\n")

    # Test
    if config['test']:
        trainer.test(
            backbone=pointr,
            model=None,
            dataloader=test_dataloader
        )
        print(f"Test loss: {trainer.test_loss:5f}")


 
if __name__ == "__main__":
    main()
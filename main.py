import os
import gc
import sys
import cv2
import json
import h5py
import yaml
import torch
import logging
from copy import deepcopy
from torch.utils.data import DataLoader
from easydict import EasyDict

# Optimizers
from torch.optim import (
    AdamW
)
from torch.optim.lr_scheduler import LambdaLR

availabel_optimizers = {
    'AdamW': AdamW,
    'LambdaLR': LambdaLR,
}

# Losses
from extensions.chamfer_dist import (
    ChamferDistanceL1, 
    ChamferDistanceL2
)
from torch.nn import MSELoss

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
    model_summary,
    dump_all_modules_parameters
)
from pointnet2_ops import pointnet2_utils

# Dataset
from mvp.mvp_dataset import MVPDataset

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra

# Models
from models.PoinTr import PoinTr
from models.PoinTrWrapper import PoinTrWrapper
from models.ga.upsampler import UpsamplingLoss
from models.ga.MVFormer import TransformerEncoderGA
from models.ga.upsampler import PointCloudUpsamplerImproved

def main():

    # Load training configuration file
    training_config = os.path.join(BASE_DIR, 'cfgs/GAPoinTr-training.yaml')
    with open(training_config, "r") as file:
        config = yaml.safe_load(file)
    # Here also select between train and test
    run_name = config['run_name']
    
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

    # Build MVP dataset for training 
    print("\nBuilding MVP Dataset...")
    logger.info(f"Datasets: {config['train_dataset']}\t{config['test_dataset']}")
    train_data_path = config['train_dataset'] 
    load_train_dataset = h5py.File(train_data_path, 'r')
    train_dataset = MVPDataset(load_train_dataset, transform_for='train', logger=logger)
    logger.info(f"lenght of train dataset: {len(train_dataset)}")
    print((f"Lenght of train dataset: {len(train_dataset)}"))

    # Build MVP dataset for testing
    test_data_path = config['test_dataset'] 
    load_test_dataset = h5py.File(test_data_path, 'r')
    test_dataset = MVPDataset(load_test_dataset, transform_for='test', logger=logger)
    logger.info(f"lenght of test dataset: {len(test_dataset)}")

    # Make dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    logger.info(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch'])
    print("done")

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
    # model_info(pointr.increase_dim)
    # model_info(pointr.reduce_map)
    # model_info(pointr.foldingnet)

    # HERE SELECT CUSTOM MODEL
    print(f"\nBuilding Custom model: {run_name}")
    # model = TransformerEncoderGA(
    #     algebra_dim = 3, 
    #     embed_dim = 8, 
    #     hidden_dim = 256, 
    #     num_layers = 2, 
    #     seq_lenght = 448,
    # )
    #model = PointCloudUpsamplerImproved(
     #   input_points=448, 
      #  output_points=2048
    #)
    model = PoinTrWrapper(pointr)
    model_info(model)
    # # Torch Info: pip install torchinfo
    # if config['debug']: 
    #     from torchinfo import summary
    #     model_summary(model)
    #     input_shape = train_dataset[42][0].shape
    #     summary(model, input_size=(config['batch_size'], *input_shape))

    # Training configuration
    model_parameters = model.parameters()
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
            'ChDL2': ChamferDistanceL2(),
            'MSE': MSELoss(),
            'UPS': UpsamplingLoss(),
        }
    }

    # Set up saving directory
    save_dir = os.path.join(
        BASE_DIR,
        config['save_path'], 
        config['pointr_config'],
    )
    run_counter = 0
    os.makedirs(save_dir, exist_ok=True)
    for file in os.listdir(save_dir):
        if run_name.split('_') == file.split('_')[:-1]: 
            run_counter += 1
    if config['overwrite_run'] and run_counter > 0: run_counter -= 1
    save_dir = os.path.join(save_dir, run_name)
    save_dir = save_dir+"_"+str(run_counter)
    logger.info(f"\nSaving checkpoints in: {save_dir}")
    logger.info(f"\nSaving losses in: {save_dir}")

    # Dump file
    if config['dump_dir']:
        dump_dir = os.path.join(config['dump_dir'], run_name)
        os.makedirs(dump_dir, exist_ok=True)
        train_dump_file = os.path.join(dump_dir, "training_dump.txt")

    # Training / Testing Class
    trainer = training.Trainer(
        parameters=trainer_parameters,
        logger=logger,
        cfg=config,
        dump_file=train_dump_file
    )

    # Train
    if config['train']:
        print("\nTraining")
        trainer.train(
            model=model,
            dataloader=train_dataloader,
            save_path=save_dir
        )
        print("\nEnd of training!\n")

    # Test
    if config['test']:
        print("Testing")
        checkpoint_file = f"{save_dir}/training/final/checkpoint.pt"
        checkpoint_file = os.path.join(BASE_DIR, '..', checkpoint_file)
        test_model = deepcopy(model)
        checkpoint = torch.load(checkpoint_file, weights_only=True)
        test_model.load_state_dict(checkpoint['model'], strict=True)

        if config['dump_dir']:
            print("Checking for difference between saved weights and loaded weights!")
            test_dump_file = os.path.join(dump_dir, "test_dump.txt")
            dump_all_modules_parameters(test_model, test_dump_file)
            difference = os.system(f"diff {test_dump_file} {train_dump_file}") 
            if difference > 0: 
                print(difference)
                raise Exception(f"Error in loading checpoint form {checkpoint_file}")       
            print("No difference found!")         

        trainer.test(
            model=test_model,
            dataloader=test_dataloader,
            save_path=save_dir
        )
        print(f"Test loss: {trainer.test_loss:5f}")


if __name__ == "__main__":
    main()



    # LOAD POINTR ALONE NO CHECKPOINTS
    ########
    # checkpoints_file = f"saves/training/PCN_models/{version}/final/checkpoint.pt"
    # checkpoints_file = os.path.join(BASE_DIR, '..', checkpoints_file)
    # pointr_config = EasyDict(
    #     {
    #         'num_pred': 14336, 
    #         'num_query': 224, 
    #         'knn_layer': 1, 
    #         'trans_dim': 384,  
    #         'ga_head': False,
    #         'ga_tail': False, 
    #         'ga_params': False
    #     }
    # )
    # model = PoinTr(pointr_config)
    # model = model.to(device)
    # gapointr_ckp = torch.load(checkpoints_file, weights_only=True)
    # model.load_state_dict(gapointr_ckp['model'])
    # gapointr_output = model(input_for_pointr)
    # output = gapointr_output[1]
    ####

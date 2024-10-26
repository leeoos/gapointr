import os
import sys
import h5py
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

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
from models.GPD import GPD



class GaTrainer():

    def __init__(self, logger) -> None:
        self.parameters = ...
        self.logger = logger

    def train(
            self,
            backbone,
            main_model,
            dataloader
    ) -> None:
        
        backbone_device = next(backbone.parameters()).device
        main_model_device = next(main_model.parameters()).device
        assert backbone_device == main_model_device
        
        for idx, pcd in enumerate(tqdm(dataloader, total=len(dataloader))):

            partial, complete = pcd

            # Pass partial pcd to PoinTr
            with torch.no_grad():
                ret = backbone(partial.to(device))

            raw_output = ret[-1].permute(0, 2, 1)
            self.logger.info(f"output shape: {raw_output.shape}")

            # Pass trough GPD
            assert main_model_device == raw_output.device
            deformed_points = main_model(raw_output) 
        



if __name__ == "__main__":

    # Setup logging
    os.makedirs(BASE_DIR + "/../logs", exist_ok=True)
    logging.basicConfig(
        filename=BASE_DIR+"/../logs/train.log", 
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
    train_dataset = MVPDataset(load_train_dataset, logger=logger)
    logger.info(f"lenght of train dataset: {len(train_dataset)}")
    print(train_dataset)
    print("done")

    # Make dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Build algebra
    points = train_dataset[42][0] # get partial pcd to build the algebra
    logger.info(f"shape of a single partial pointcloud: {points[0].shape}")
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
    pointr = builder.model_builder(config.model)
    builder.load_model(pointr, pointr_ckp)
    pointr.to(device)

    # Define custom model
    # model = PointCloudDeformationNet()
    print("\nBuilding the GPD model")
    gpd = GPD(algebra)
    gpd.to(device)
    param_device = next(gpd.parameters()).device
    logger.info(f"model parameters device: {param_device}") 
    print(gpd.name)
    print(gpd)
    print("done")

    trainer = GaTrainer(
        logger=logger
    )

    trainer.train(
        backbone=pointr,
        main_model=gpd,
        dataloader=train_dataloader
    )

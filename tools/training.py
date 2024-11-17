import os
import gc
import sys
import cv2
import h5py
import yaml
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import (
    AdamW
)
availabel_optimizers = {
    'AdamW': AdamW
}

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# PoinTr imports
from utils import misc
from tools import builder
from utils.config import cfg_from_yaml_file,  get_instance
from extensions.chamfer_dist import (
    ChamferDistanceL1, 
    ChamferDistanceL2
)

# MVP Dataset  
from mvp.mvp_dataset import MVPDataset

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra

# Models
from models.ga_models.GAFold import GAFold
from models.ga_models.MVFormer import SelfAttentionGA


class Trainer():

    def __init__(self, parameters, algebra, logger) -> None:
        self.parameters = parameters
        self.algebra = algebra
        self.logger = logger


    def train(self, backbone, model, dataloader) -> None:
        backbone_device = next(backbone.parameters()).device
        main_model_device = next(model.parameters()).device
        assert backbone_device == main_model_device

        flush_step = 500
        epoch_loss = None
        device = self.parameters['device']
        train_epochs = self.parameters['epochs']
        optimizer = self.parameters['optimizer']
        losses = self.parameters['losses']

        # To refine 
        loss_fn1 = losses['ChDL1']
        loss_fn2 = losses['ChDL2']

        print("\nLosses:")
        print(loss_fn1)
        print(loss_fn2)
        print("\nOptimizer:")
        print(optimizer)
    
        with tqdm(total=train_epochs, leave=True) as pbar:
            for epoch in range(train_epochs):
                epoch_loss = 0
                with tqdm(total=len(dataloader)) as bbar:
                    for step, pcd in enumerate(dataloader):

                        partial, complete = pcd
                        optimizer.zero_grad()

                        # Send point clouds to device
                        partial = partial.to(device)
                        complete = complete.to(torch.float32).to(device)

                        # Pass partial pcd to PoinTr
                        pointr_parameters = {}
                        with torch.no_grad():
                            pointr_parameters = backbone(partial)[-1]

                        # Pass trough GA model
                        output = model(partial, backbone, pointr_parameters)

                        loss = loss_fn1(output, complete) + loss_fn2(output, complete)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                        bbar.set_postfix(
                            batch_loss= loss.item(), epoch=(epoch/(step + 1))*100
                        )
                        bbar.update(1)

                        # free up cuda mem
                        del complete
                        del partial
                        del output
                        del loss
                        if step > 0 and step % flush_step == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                            gc.collect()
            
                epoch_loss = epoch_loss/(step + 1)
                pbar.set_postfix(epoch_train=epoch_loss)
                pbar.update(1)
                
def main():

    # Load training configuration file
    training_config = os.path.join(BASE_DIR, '../cfgs/GAPoinTr-training.yaml')
    with open(training_config, "r") as file:
        config = yaml.safe_load(file)

    # Saving path
    save_dir = os.path.join(
        BASE_DIR, '..', config['save_path'], config['pointr_config'].lower()
    )
    
    # Setup logging
    os.makedirs(BASE_DIR + "../logs", exist_ok=True)
    logging.basicConfig(
        filename=BASE_DIR + "../logs/train.log", 
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

    # Make dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"Number of batches: {len(train_dataloader)}")
    print("done")

    # Build algebra
    points = train_dataset[42][0] # get partial pcd to build the algebra
    logger.info(f"shape of a single partial pointcloud: {points[0].shape}")
    algebra_dim = int(points.shape[1])
    metric = [1 for i in range(algebra_dim)]
    print("\nGenerating the algebra...")
    algebra = CliffordAlgebra(metric)
    print(f"algebra dimention: \t {algebra.dim}")
    print(f"multivectors elements: \t {sum(algebra.subspaces)}")
    print(f"number of subspaces: \t {algebra.n_subspaces}")
    print(f"subspaces grades: \t {algebra.grades.tolist()}")
    print(f"subspaces dimentions: \t {algebra.subspaces.tolist()}")
    print("done")

    # Define PoinTr instance
    print("\nBuilding PoinTr...")
    pointr_init_config = os.path.join(
        BASE_DIR, "../cfgs", config['pointr_config'], "PoinTr.yaml"
    )
    pointr_ckp = os.path.join(BASE_DIR, "../ckpts", config['pointr_config'], "pointr.pth")
    pointr_config = cfg_from_yaml_file(pointr_init_config, root=BASE_DIR+"/../")
    pointr = builder.model_builder(pointr_config.model)
    builder.load_model(pointr, pointr_ckp)
    pointr = pointr.to(device)

    # Define custom model
    print("\nBuilding GAPoinTr...")
    model = GAFold(
        algebra=algebra,  
        embed_dim=config['embed_dim']
    )
    model = model.to(device)
    param_device = next(model.parameters()).device
    logger.info(f"model parameters device: {param_device}") 

    # GAPoinTr parametr estimation
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = total_params * 4  # Assuming float32
    model_size_mb = param_size_bytes / (1024 ** 2)
    print(f"Total Parameters: {total_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print("done")

    # Build parameters from yaml
    parameters = {
        'epochs': config['epochs'],
        'optimizer': get_instance(
                        config['optimizer'], 
                        availabel_optimizers,
                        {"params": model.parameters()}
                    ),
        'device': device,
        'losses':  {
            'ChDL1': ChamferDistanceL1(),
            'ChDL2': ChamferDistanceL2()
        }
    }

    # Training
    print("\nTraining")
    trainer = Trainer(
        parameters=parameters,
        algebra=algebra,
        logger=logger
    )
    trainer.train(
        backbone=pointr,
        model=model,
        dataloader=train_dataloader
    )

    # Saving train output
    if config['override_cache']:
        run_counter = config['run_counter']
        run_counter = str(int(run_counter) + 1)
        save_dir = os.path.join(save_dir, "_"+run_counter)
        config['run_counter'] = run_counter
        with open(training_config, "w") as cfile:
            yaml.dump(config, cfile)
        os.makedirs(save_dir, exist_ok=True)

    else:
        os.makedirs(save_dir, exist_ok=True)

    save_file = os.path.join(save_dir, "model_state_dict.pt")
    print(f"\nSaving checkpoints in: {save_dir}")
    torch.save(
        model.state_dict(), 
        save_file
    )
    
if __name__ == "__main__":
    main()
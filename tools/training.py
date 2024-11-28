import os
import gc
import sys
import cv2
import json
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
from pointnet2_ops import pointnet2_utils

# # MVP Dataset  
# from mvp.mvp_dataset import MVPDataset

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra


class Trainer():

    def __init__(
        self, 
        parameters, 
        logger, 
        debug=False, 
    ) -> None:
        
        self.parameters = parameters
        self.logger = logger
        self.debug = debug

        self.loss_trend = {}
        self.test_loss = 0 

    def train(self, backbone, model, dataloader, save_path) -> None:
        # backbone_device = next(backbone.parameters()).device
        # main_model_device = next(model.parameters()).device
        # assert backbone_device == main_model_device

        flush_step = 500
        epoch_loss = None
        device = self.parameters.get('device', None)
        train_epochs = self.parameters.get('epochs', None)
        optimizer = self.parameters.get('optimizer', None)
        scheduler = self.parameters.get('scheduler', None)
        losses = self.parameters.get('losses', None)

        # To refine 
        loss_fn = losses['ChDL1']

        print("\nAvailable Losses:")
        print(losses)

        print("\nOptimizer:")
        print(optimizer)
    
        with tqdm(total=train_epochs, leave=True, disable=self.debug) as pbar:
            for epoch in range(train_epochs):
                epoch_loss = 0
                with tqdm(total=len(dataloader), disable=self.debug) as bbar:
                    for step, pcd in enumerate(dataloader):

                        partial, complete = pcd
                        optimizer.zero_grad()

                        # Send point clouds to device
                        partial = partial.to(device)
                        complete = complete.to(torch.float32).to(device)

                        # Pass partial pcd to PoinTr
                        pointr_parameters = {}
                        # with torch.no_grad():
                        output = backbone(partial)
                        # pointr_parameters = pointr_output[-1]

                        # # Pass trough GA model
                        # output = model(backbone, pointr_parameters)

                        # loss = loss_fn(output, complete) #+ loss_fn2(output, complete)
                        # loss = loss_fn(output, complete) + loss_fn(pointr_output[0], complete)
                        # loss = loss_fn(output[1], complete) + loss_fn(output[0], complete)
                        loss = loss_fn(output[0], complete) # optimize only coarse
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                        bbar.set_postfix(batch_loss=loss.item())
                        bbar.update(1)

                        # collect epoch losses for statistic
                        epoch_key = "epoch" + "_" + str(epoch)
                        if  epoch_key not in self.loss_trend.keys():
                            self.loss_trend[epoch_key] = [loss.item()]
                        else:
                            self.loss_trend[epoch_key].append(loss.item())

                        # free up cuda mem
                        del pointr_output
                        del complete
                        del partial
                        del output
                        del loss

                        # Save and flush
                        if step > 0 and step % flush_step == 0:

                            # Save
                            if not self.debug:
                                checkpoint = { 
                                    'epoch': epoch,
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                }
                                save_dir=  os.path.join(
                                    save_path, f"{str(step*(epoch+1))}"
                                )
                                os.makedirs(save_dir, exist_ok=True)
                                save_file = os.path.join(save_dir, "checkpoint.pt")
                                torch.save(checkpoint, save_file)

                            # Flush
                            gc.collect()
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                        if self.debug: break

                if scheduler: scheduler.step()
                epoch_loss = epoch_loss/(step + 1)
                pbar.set_postfix(epoch_train=epoch_loss)
                pbar.update(1)
                if self.debug: break

                # Final save
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_dir=  os.path.join(
                    save_path, "final"
                )
                os.makedirs(save_dir, exist_ok=True)
                save_file = os.path.join(save_dir, "checkpoint.pt")
                torch.save(checkpoint, save_file)
                save_loss = os.path.join(save_dir, "train_losses.json")
                with open(save_loss, "w") as l_file:
                    json.dump(self.loss_trend, l_file, indent=4)


    def test(self, backbone, model, dataloader):

        flush_step = 500
        batch_loss = 0
        device = self.parameters['device']
        losses = self.parameters['losses']
        loss_fn = losses['ChDL1']

        # model.eval()
        backbone.eval()

        with tqdm(total=len(dataloader)) as bbar:
            for step, pcd in enumerate(dataloader):

                partial, complete = pcd

                # Send point clouds to device
                partial = partial.to(device)
                complete = complete.to(torch.float32).to(device)

                # Pass partial pcd to PoinTr
                pointr_parameters = {}
                with torch.no_grad():
                    output = backbone(partial)
                    # pointr_parameters = pointr_output[-1]
                    # output = model(backbone, pointr_parameters)

                loss = loss_fn(output[-1], complete)
                batch_loss += loss.item()

                bbar.set_postfix(batch_loss=loss.item())
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

            self.test_loss = batch_loss/(step + 1)
            
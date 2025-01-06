import os
import gc
import sys
import json
import torch
import inspect
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import (
    AdamW
)
availabel_optimizers = {
    'AdamW': AdamW
}
torch.manual_seed(42)

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# PoinTr imports
from utils import misc
from tools import builder
from utils.config import (
    cfg_from_yaml_file,  
    get_instance, 
    dump_all_modules_parameters
)

# Metrics
from utils.metrics import Metrics
from utils.dist_utils import fps
from pointnet2_ops import pointnet2_utils

# # MVP Dataset  
# from mvp.mvp_dataset import MVPDataset

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra


class Trainer():

    def __init__(self, parameters, logger, cfg, dump_file='' ) -> None:

        self.debug = cfg['debug']
        self.resume = cfg['resume']
        self.b_size = cfg['batch_size']
        self.run_name = cfg['run_name']
        self.save_step = cfg['save_step']
        self.pretrained = cfg['pretrained']
        self.resume_step = cfg['resume_step']
        self.progressive = cfg['progressive_saves']
        self.accumulation_step = cfg['accumulation_step']

        self.parameters = parameters
        self.dump_file = dump_file
        self.logger = logger

        self.total_epochs = 0
        self.total_steps = 0
        self.loss_trend = {}
        self.test_loss = 0 

    def train(self, model, dataloader, save_path) -> None:

        flush_step = 500
        epoch_loss = None
        device = self.parameters.get('device', None)
        train_epochs = self.parameters.get('epochs', None)
        optimizer = self.parameters.get('optimizer', None)
        scheduler = self.parameters.get('scheduler', None)
        loss_fn = model.train_loss

        # Ensure device
        model = model.to(device)

        # Info
        train_loss_signature = inspect.getsource(loss_fn)
        train_loss_signature = train_loss_signature.split('=')[1].strip()
        print(f"Train Loss Function: {train_loss_signature}")
        print("Optimizer:")
        pprint(optimizer)

        if self.resume and self.total_steps < self.resume_step:
            print(f"Skipping the first {self.resume_step}")
            self.logger.info(f"Skipping the first {self.resume_step}")
    
        with tqdm(total=train_epochs, leave=True, disable=self.debug) as pbar:
            for epoch in range(train_epochs):
                epoch_loss = 0
                with tqdm(total=len(dataloader), leave=False, disable=self.debug) as bbar:
                    for step, pcd in enumerate(dataloader):
                        
                        if self.resume and  self.total_steps < self.resume_step:
                            self.total_steps += 1
                            bbar.update(1)
                            continue

                        # Get samples and reset optimizer
                        partial, complete = pcd
                        optimizer.zero_grad()

                        # Send point clouds to device
                        partial = partial.to(device)
                        complete = complete.to(torch.float32).to(device)
                        output = model(partial)

                        # Backward step
                        loss = loss_fn(output, complete)
                        loss = loss / self.accumulation_step
                        loss.backward()

                        # Wait for several backward steps for gradient accumulation
                        if (step + 1) % self.accumulation_step == 0: 
                            optimizer.step()                            
                            model.zero_grad() 
                            optimizer.zero_grad()

                        epoch_loss += loss.item()
                        # optimizer.step()

                        if self.debug: print(f"loss: {loss.item()}")
                        bbar.set_postfix(batch_loss=loss.item())
                        bbar.update(1)
                        self.logger.info(f"batch loss: {loss.item()}\tepoch: {step/len(dataloader)}")

                        # Collect epoch losses for statistic
                        epoch_key = "epoch" + "_" + str(epoch+1)
                        if  epoch_key not in self.loss_trend.keys():
                            self.loss_trend[epoch_key] = [loss.item()]
                        else:
                            self.loss_trend[epoch_key].append(loss.item())

                        # Save and flush
                        if self.total_steps > 0 and self.total_steps % self.save_step == 0 and\
                            self.progressive and not self.debug:
                                checkpoint = { 
                                    'epoch': epoch,
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                }
                                save_dir=  os.path.join(
                                    save_path, f"training/{str(self.total_steps)}"
                                )
                                os.makedirs(save_dir, exist_ok=True)
                                save_file = os.path.join(save_dir, "checkpoint.pt")
                                torch.save(checkpoint, save_file)
                        
                        # Free up cuda mem
                        del complete
                        del partial
                        del output
                        del loss

                        # Flush
                        if step > 0 and step % flush_step == 0:
                            # Flush
                            bbar.set_postfix(flush="flush")
                            gc.collect()
                            torch.cuda.empty_cache()
                            gc.collect()

                        self.total_steps += 1
                        if self.debug: break

                if scheduler: scheduler.step()
                epoch_loss = epoch_loss/(step + 1)
                pbar.set_postfix(epoch_train=epoch_loss)
                pbar.update(1)
                if self.debug: break

            # Final save
            self.total_epochs = epoch + 1
            if not self.debug:
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_dir=  os.path.join(
                    save_path, f"training/{self.total_steps}"
                )
                os.makedirs(save_dir, exist_ok=True)
                save_file = os.path.join(save_dir, "checkpoint.pt")
                torch.save(checkpoint, save_file)
                save_loss = os.path.join(save_dir, "train_losses.json")
                with open(save_loss, "w") as l_file:
                    json.dump(self.loss_trend, l_file, indent=4)


            # Dump model state after training
            if self.dump_file:
                dump_all_modules_parameters(model, self.dump_file)
                

    def test(self, model, dataloader, save_path):

        flush_step = 500
        batch_loss = 0
        device = self.parameters['device']
        train_loss_fn = model.train_loss
        loss_fn = model.test_loss

        model = model.to(device)
        model.eval()

        self.logger.info(f"Testing...")
        train_loss_signature = inspect.getsource(train_loss_fn)
        train_loss_signature = train_loss_signature.split('=')[1].strip()
        test_loss_signature = inspect.getsource(loss_fn)
        test_loss_signature = test_loss_signature.split('=')[1].strip()
        print(f"Test Loss Function: {test_loss_signature}")

        evaluation_results = {}
        results_file_name = "experiments.json" if self.pretrained else "full_results.json"
        results_file = os.path.join(BASE_DIR, "..", f"results/metrics/{results_file_name}")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with tqdm(total=len(dataloader), disable=self.debug) as bbar:
            for step, pcd in enumerate(dataloader):

                partial, complete = pcd

                # Send point clouds to device
                partial = partial.to(device)
                complete = complete.to(torch.float32).to(device)

                # Pass partial pcd to PoinTr
                with torch.no_grad():
                    output = model(partial)

                loss = loss_fn(output, complete)
                batch_loss += loss.item()

                self.evaluation(output[-1], complete, evaluation_results)

                bbar.set_postfix(batch_loss=loss.item())
                bbar.update(1)
                self.logger.info(f"test loss: {loss.item()}")

                # free up cuda mem
                del complete
                del partial
                del output
                # del loss
                if step > 0 and step % flush_step == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.debug: break
                # if step == 5: break
        
        results_dic = {}
        try:
            with open(results_file, "r") as rfile:
                results_dic = json.load(rfile)
        except:
            results_dic = {}
        if self.run_name not in results_dic.keys(): results_dic[self.run_name] = {}
                
        # Print test results     
        print("\nEvaluation")
        self.test_loss = batch_loss/(step + 1)
        print(f"Test loss: {self.test_loss:5f}")
        for key in evaluation_results.keys():
            evaluation_results[key] /= (step + 1)
            print(f"{key}: {evaluation_results[key]:5f}")
            results_dic[self.run_name][key] = evaluation_results[key]

        if not self.debug:
            print("\nSaving results...")
            print(f"Global file for results summary: {results_file}")
            self.logger.info(f"Global file for results summary: {results_file}")
            with open(results_file, "w") as rout:
                json.dump(results_dic, rout, indent=4)
        
            save_dir = os.path.join(save_path, "evaluation/")
            os.makedirs(save_dir, exist_ok=True)
            if self.total_epochs > 0: 
                with open(os.path.join(save_dir, "run_info.txt"), "w") as file:
                    file.write(f"Run name: {self.run_name}\n")
                    file.write(f"Using pretrained: {self.pretrained}\n")
                    file.write(f"Training epochs: {self.total_epochs}\n")
                    file.write(f"Training batch size: {self.b_size}\n")
                    file.write(f"Optimizer: {self.parameters.get('optimizer', None)}\n")
                    file.write(f"Train Loss Function: {train_loss_signature}\n")
                    file.write(f"Test Loss Function: {test_loss_signature}\n")
                    file.write(f"Test Loss: {str((self.test_loss))}\n")
                    for index, items in enumerate(evaluation_results.items()):
                        key, value = items
                        file.write(f"{key}: {value}")
                        if (index < len(evaluation_results) - 1): file.write('\n')
            else:
                try:
                    with open(os.path.join(save_dir, "run_info.txt"), "r+") as file:
                        lines = file.readlines()
                        lines = lines[:-(len(evaluation_results)+2)]
                        file.seek(0)
                        file.writelines(lines)
                        file.write(f"Test Loss Function: {test_loss_signature}\n")
                        file.write(f"Test Loss: {str((self.test_loss))}\n")
                        for index, items in enumerate(evaluation_results.items()):
                            key, value = items
                            file.write(f"{key}: {value}")
                            if (index < len(evaluation_results) - 1): file.write('\n')
                except:
                    with open(os.path.join(save_dir, "run_info.txt"), "w") as file:
                        file.write(f"Run name: {self.run_name}\n")
                        file.write(f"Test Loss Function: {test_loss_signature}\n")
                        file.write(f"Test Loss: {str((self.test_loss))}\n")
                        for index, items in enumerate(evaluation_results.items()):
                            key, value = items
                            file.write(f"{key}: {value}")
                            if (index < len(evaluation_results) - 1): file.write('\n')

        print("done!")

    def evaluation(self, prediction, target, evaluation_results):
        to_test = fps(prediction.detach(), num=2048)
        metrics = Metrics.get(to_test.detach(), target, require_emd=True)
        metric_names = Metrics.names()
        for name, value in zip(metric_names, metrics):
            evaluation_results.setdefault(name, 0)
            evaluation_results[name] += value.item()




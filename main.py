import os
import sys
import h5py
import yaml
import torch
import logging
from copy import deepcopy
from easydict import EasyDict
from torch.utils.data import DataLoader
torch.manual_seed(42)

# Optimizers
from torch.optim import (
    AdamW
)
from torch.optim.lr_scheduler import LambdaLR
availabel_optimizers = {
    'AdamW': AdamW,
    'LambdaLR': LambdaLR,
}

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

def main():

    # Load training configuration file
    training_config = os.path.join(BASE_DIR, 'cfgs/GAPoinTr-training.yaml')
    with open(training_config, "r") as file:
        config = yaml.safe_load(file)
    run_name = config['run_name']
    print(f"Experiment: {run_name}")
    
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
    test_dataset = MVPDataset(load_test_dataset, transform_for='train', logger=logger)
    logger.info(f"lenght of test dataset: {len(test_dataset)}")

    # Make dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    logger.info(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch'])
    print("done")

    # Set up saving directory
    save_dir = os.path.join(
        BASE_DIR,
        config['save_path'], 
        config['pointr_config'],
        "fine-tuning" if config['pretrained'] else "full"
    )
    run_counter = 0
    if not config['debug']: os.makedirs(save_dir, exist_ok=True)
    for file in os.listdir(save_dir):
        if run_name.split('_') == file.split('_')[:-1]: 
            run_counter += 1
    if (config['overwrite_run'] or config['resume']) and run_counter >= 1: 
        run_counter -= 1
    # Use predefined version if specified
    run_counter = config['version_number'] if isinstance(config['version_number'], int) else run_counter
    run_name += "_" + str(run_counter)
    config['run_name'] = run_name
    save_dir = os.path.join(save_dir, run_name)
    print(f"\nSaving point: {save_dir}")
    logger.info(f"\nSaving point: {save_dir}")

    # Preloaded optimizer initialization
    selected_optimizer = None

    # Define PoinTr instance
    print("\nBuilding PoinTr...")
    if config['pretrained']:
        pointr_init_config = os.path.join(
            BASE_DIR, "cfgs", config['pointr_config'], "PoinTr.yaml"
        )
        pointr_ckp = os.path.join(BASE_DIR, "ckpts", config['pointr_config'], "pointr.pth")
        pointr_config = cfg_from_yaml_file(pointr_init_config, root=BASE_DIR+"/")
        pointr = builder.model_builder(pointr_config.model)
        builder.load_model(pointr, pointr_ckp)
        pointr = pointr.to(device)

        # Load optimizer state
        if config['load_optimizer']:
            selected_optimizer = AdamW(pointr.parameters())
            raw_ckps = torch.load(pointr_ckp, weights_only=True)
            selected_optimizer.load_state_dict(raw_ckps['optimizer'])

    else:
        pointr_config = EasyDict(
            {
                'num_pred': 14336, 
                'num_query': 224, 
                'knn_layer': 1, 
                'trans_dim': 384,  
            }
        )
        pointr = PoinTr(pointr_config)
    model_info(pointr)

    print(f"\nBuilding Custom model: {run_name}")
    model = PoinTrWrapper(pointr=pointr, gafet=config['gafet'])
    model_info(model)
    # # Torch Info: pip install torchinfo --> commit new docker --> docker commit <container id> pointr-ga:configured
    # if config['debug']: 
    #     from torchinfo import summary
    #     model_summary(model)
    #     input_shape = train_dataset[42][0].shape
    #     summary(model, input_size=(config['batch_size'], *input_shape))

    if config['resume'] and config['train']:
        try:
            resume_step = []
            for file in os.listdir(os.path.join(save_dir, 'training')):
                if file != 'final': resume_step.append(int(file))
            resume_step = max(resume_step)
            config['resume_step'] = int(resume_step)
            
            # Load model parameters
            resume_checkpoint_file = f"{save_dir}/training/{resume_step}/checkpoint.pt"
            print(f"Loading model from {resume_checkpoint_file}")
            resume_checkpoint = torch.load(resume_checkpoint_file, weights_only=True)
            model.load_state_dict(resume_checkpoint['model'], strict=False) # RUN YOU CLEVER BOY AND REMEMBER
            model = model.to(device)
            print("Loading optimizer...")
            selected_optimizer = AdamW(pointr.parameters())
            selected_optimizer.load_state_dict(resume_checkpoint['optimizer'])

        except:
            print(f"\nImpossible to resume training from: {save_dir}")
            print(f"Restart training")
            config['resume'] = False
            config['resume_step'] = 1_000_000_000
            pass
    else:
        config['resume_step'] = 1_000_000_000

            
    # Training configuration
    model_parameters = model.get_model_parameters()
    trainer_parameters = {
        'epochs': config['epochs'],
        'optimizer': get_instance(
                        config['optimizer'], 
                        availabel_optimizers,
                        {"params": model_parameters}
                    ) if not selected_optimizer else selected_optimizer,
        'scheduler': None,
        'device': device,
    }

    # Dump file
    if config['dump_dir']:
        dump_dir = os.path.join(config['dump_dir'], run_name)
        os.makedirs(dump_dir, exist_ok=True)
        train_dump_file = os.path.join(dump_dir, "training_dump.txt")
    else:
        train_dump_file = None

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
        final = []
        try:
            for file in os.listdir(os.path.join(save_dir, 'training')):
                if file != 'final': final.append(int(file))
            final = max(final)
        except Exception as error:
            print(f"Impossible to test without a valid checkpoint --> {error}")

        print(f"Loading verasion: {save_dir}/training/{final}/checkpoint.pt")
        logger.info(f"Loading verasion: {save_dir}/training/{final}/checkpoint.pt")
        checkpoint_file = f"{save_dir}/training/{final}/checkpoint.pt"
        checkpoint_file = os.path.join(BASE_DIR, '..', checkpoint_file)
        test_model = deepcopy(model)

        if not config['debug'] and config['load_ckp']:
            checkpoint = torch.load(checkpoint_file, weights_only=True)
            test_model.load_state_dict(checkpoint['model'], strict=True)


        if config['dump_dir'] and config['train']:
            print("Checking for difference between saved weights and loaded weights!")
            test_dump_file = os.path.join(dump_dir, "test_dump.txt")
            dump_all_modules_parameters(test_model, test_dump_file)
            difference = os.system(f"diff {test_dump_file} {train_dump_file}") 
            if difference > 0: 
                print(difference)
                raise Exception(f"Error in loading checpoint form {checkpoint_file}")       
            print(f"from {dump_dir} --> No difference found!")         

        trainer.test(
            model=test_model,
            dataloader=test_dataloader,
            save_path=save_dir
        )

if __name__ == "__main__":
    main()


import os
import gc
import sys
import cv2
import h5py
import torch
import logging
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# PoinTr imports
from utils import misc
from tools import builder
from utils.config import cfg_from_yaml_file
from extensions.chamfer_dist import ChamferDistanceL1

# MVP Clifford Algebra, Model, Dataset  
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra
from mvp.mvp_dataset import MVPDataset
from models.ga_models.GPD import InvariantCGENN
from models.ga_models.CGNN import CGNN


class GaTrainer():

    def __init__(self, parameters, algebra, logger) -> None:
        self.parameters = parameters
        self.algebra = algebra
        self.logger = logger

    # Chamfer Distance helper function
    def chamfer_distance(self, point_cloud1, point_cloud2):
        dist1 = torch.cdist(point_cloud1, point_cloud2, p=2).min(dim=1)[0]
        dist2 = torch.cdist(point_cloud2, point_cloud1, p=2).min(dim=1)[0]
        return dist1.mean() + dist2.mean()

    def train(
            self,
            backbone,
            main_model,
            dataloader
    ) -> None:
        
        backbone_device = next(backbone.parameters()).device
        main_model_device = next(main_model.parameters()).device
        assert backbone_device == main_model_device

        epoch_loss = None
        train_epochs = self.parameters['epochs']
        optimizer = self.parameters['optimizer']
        loss_fn = self.parameters['loss']
    
        with tqdm(total=train_epochs, leave=True) as pbar:
            for epoch in range(train_epochs):
                epoch_loss = 0
                stop_at = 5
                for batch_idx, pcd in enumerate(dataloader):

                    partial, complete = pcd
                    optimizer.zero_grad()

                    # Pass partial pcd to PoinTr
                    with torch.no_grad():
                        output = backbone(partial.to(device))[-1]

                    target = torch.tensor(complete, dtype=torch.float32).unsqueeze(0).to(device)
                    # self.logger.info(f"output shape: {raw_output.shape}")

                    # Pass trough GPD
                    assert main_model_device == output.device
                    refined_points = main_model(output) 

                    loss = self.chamfer_distance(refined_points, target) #loss_fn(refined_points, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    # free up cuda mem
                    del partial
                    del complete
                    del target
                    del refined_points
                    del loss
                    gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()

                    if batch_idx == stop_at:
                        break
            
                epoch_loss = epoch_loss/(batch_idx + 1)
                pbar.set_postfix(train=epoch_loss)
                pbar.update(1)
                


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
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

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
    init_config = BASE_DIR + "/../cfgs/PCN_models/PoinTr.yaml"
    pointr_ckp = BASE_DIR + "/../ckpts/PCN_Pretrained.pth"
    config = cfg_from_yaml_file(init_config, root=BASE_DIR+"/../")
    pointr = builder.model_builder(config.model)
    builder.load_model(pointr, pointr_ckp)
    pointr = pointr.to(device)

    # Define custom model
    # model = PointCloudDeformationNet()
    print("\nBuilding the Refiner model")
    cgnn = CGNN(algebra)
    cgnn = cgnn.to(device)
    param_device = next(cgnn.parameters()).device
    logger.info(f"model parameters device: {param_device}") 
    print("done")

    print("\nTraining")
    parameters = {
        'epochs': 3,
        'optimizer': optim.AdamW(cgnn.parameters(), lr=0.0001),
        'loss': torch.nn.MSELoss() #ChamferDistanceL1()
    }
    trainer = GaTrainer(
        parameters=parameters,
        algebra=algebra,
        logger=logger
    )

    trainer.train(
        backbone=pointr,
        main_model=cgnn,
        dataloader=train_dataloader
    )

    output_dir = BASE_DIR + "/../inference_result/training/"
    os.makedirs(output_dir, exist_ok=True)

    # Build MVP testset
    print("\nBuilding MVP Testset...")
    data_path = BASE_DIR + "/../mvp/datasets/"
    train_data_path = data_path + "MVP_Test_CP.h5"
    logger.info(f"data directory: {data_path}")
    load_train_dataset = h5py.File(train_data_path, 'r')
    train_dataset = MVPDataset(load_train_dataset, logger=logger, mv=26)
    logger.info(f"lenght of test dataset: {len(train_dataset)}")
    print(train_dataset)
    print("done")

    # Temporary get a single sample 
    pcd_index = 20
    partial, complete = train_dataset[pcd_index]
    input_img = misc.get_ptcloud_img(partial)
    complete_img = misc.get_ptcloud_img(complete)
    cv2.imwrite(os.path.join(output_dir, 'partial.jpg'), input_img)
    cv2.imwrite(os.path.join(output_dir, 'complete.jpg'), complete_img)
    logger.info(f"shape of a single partial pointcloud: {partial[0].shape}")

    # Pipeline
    print("\nPipeline inference...")
    pointr.eval()
    input_for_pointr = torch.tensor(partial, dtype=torch.float32).unsqueeze(0).to(device)
    ret = pointr(input_for_pointr)
    raw_output = ret[-1] #.permute(1, 2, 0)
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    dense_img = misc.get_ptcloud_img(dense_points)
    cgnn.eval()
    deformed_points = cgnn(raw_output)
    ga_dense_points = deformed_points.squeeze(0).detach().cpu().numpy()
    cv2.imwrite(os.path.join(output_dir, 'fine.jpg'), dense_img)
    ga_img = misc.get_ptcloud_img(ga_dense_points)
    cv2.imwrite(os.path.join(output_dir, 'ga_fine.jpg'), ga_img)

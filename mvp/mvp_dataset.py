import os 
import sys
import math
import random 
from traitlets import Any
from torch.utils.data import Dataset

# random.seed(346)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import utils.data_transforms as data_transforms


class MVPDataset(Dataset):

    def __init__(self, h5_dataset, transform_for='', mv=26, logger=None) -> None:
        super().__init__()
        self.name = "mvp_datasets"
        self.dataset = h5_dataset
        self.mv = mv # number of multiview instances of partial pcd for each complete pcd

        # print(len(self.dataset['complete_pcds']))
        # print(len(self.dataset['incomplete_pcds']))
        # print(len(self.dataset['labels']))
        # exit()

        if logger:
            logger.info(f"len of complete pcd: {len(self.dataset['complete_pcds'])}")
            logger.info(f"len of incomplete pcd: {len(self.dataset['incomplete_pcds'])}")

        assert len(self.dataset['incomplete_pcds']) == mv * len(self.dataset['complete_pcds']), """The number of partial pointclouds
        should match the number of complete one"""

        # transform should be either: 'train' or 'test'
        if transform_for:
            self.transform = self._get_transforms(transform_for)
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.dataset['incomplete_pcds'])

    def _get_transforms(self, transform_for):

        if transform_for == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        
    def __get_class__(self, index):
        # target_index = math.floor(index / self.mv)
        return self.dataset['labels'][index]
    
    def __getitem__(self, index) -> Any:

        target_index = math.floor(index / self.mv) # crop the index for the complete pcd
        # print(index)
        # print(target_index)

        partial = self.dataset['incomplete_pcds'][index]
        complete = self.dataset['complete_pcds'][target_index]

        data = {
            'partial': partial,
            'gt': complete
        }

        if self.transform:
            data = self.transform(data)

        return (data['partial'], data['gt']) 
    
    
    
import math
import random 
from traitlets import Any
from torch.utils.data import Dataset

random.seed(346)


class MVPDataset(Dataset):

    def __init__(self, h5_dataset, mv=26, logger=None) -> None:
        super().__init__()
        self.name = "mvp_datasets"
        self.dataset = h5_dataset
        self.mv = mv # number of multiview instances of partial pcd for each complete pcd

        if logger:
            logger.info(f"len of complete pcd: {len(self.dataset['complete_pcds'])}")
            logger.info(f"len of incomplete pcd: {len(self.dataset['incomplete_pcds'])}")

        assert len(self.dataset['incomplete_pcds']) == mv * len(self.dataset['complete_pcds']), """The number of partial pointclouds
        should match the number of complete one"""

    def __len__(self) -> int:
        return len(self.dataset['complete_pcds'])

    def __getitem__(self, index) -> Any:

        target_index = math.floor(index / self.mv) # crop the index for the complete pcd
        # print(index)
        # print(target_index)

        partial = self.dataset['incomplete_pcds'][index]
        complete = self.dataset['complete_pcds'][target_index]

        return partial, complete    
    
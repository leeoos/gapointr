from torch.utils.data import Dataset
from traitlets import Any


class MVPDataset(Dataset):

    def __init__(self, h5_dataset) -> None:
        super().__init__()
        self.name = "mvp_datasets"
        self.dataset = h5_dataset
        assert len(self.dataset['complete_pcds']) == len(self.dataset['incomplete_pcds']), """The number of partial pointclouds
        should match the number of complete one"""

    def __len__(self) -> int:
        return len(self.dataset['complete_pcds'])

    def __getitem__(self, index) -> Any:

        partial = self.dataset['incomplete_pcds'][index]
        complete = self.dataset['complete_pcds'][index]

        return partial, complete    
    
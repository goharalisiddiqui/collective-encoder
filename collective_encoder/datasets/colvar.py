import numpy as np

from torch.utils.data import Dataset
from collective_encoder.datasets.base import BaseDataset

class ColvarData(Dataset, BaseDataset):
    """COLVAR dataset"""

    def __init__(self, 
                 frames:  np.ndarray,
                 labels: np.ndarray,
                 **kwargs):
        super().__init__(**kwargs)
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = self.frames[idx]
        y = self.labels[idx]
        return x, y

    def get_norm_data(self):
        """Return data for normalization purposes."""
        return self.frames

    def get_datapoint_shape(self):
        """Return shape of a single datapoint."""
        return self.frames[0].shape
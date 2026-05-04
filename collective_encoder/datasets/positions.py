import os

from typing import Dict, List, Union

import numpy as np
import ase

import torch
from torch.utils.data import Dataset

from .base import BaseDataset


class PositionsDataset(Dataset, BaseDataset):
    """ Dataset for atomic positions."""
    
    _IDENTIFIER = "POSITIONS"
    _REQUIRED_ARGS = []
    _OPTIONAL_ARGS = {}

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[List[float]],
        dataset_args: Dict[str, Union[float, int, str]] = None,
        **kwargs,
    ):
        Dataset.__init__(self)
        BaseDataset.__init__(self, dataset_args=dataset_args, **kwargs)
        
        self.positions = [torch.tensor(s.positions) for s in structures]
        self.labels = [torch.tensor(l).flatten() for l in labels]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        x = ()
        x += (self.positions[index],self.labels[index])
        return x
    
    def get_data(self):
        return np.array([d.numpy() for d in self.positions]), np.array([l.numpy() for l in self.labels])
    
    def get_norm_data(self) -> np.ndarray:
        return np.array([d.numpy() for d in self.positions])

    def get_datapoint_shape(self) -> tuple:
        return tuple(self.positions[0].shape)

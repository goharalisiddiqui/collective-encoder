import os

from typing import List

import numpy as np
import ase

import torch
from torch.utils.data import Dataset


class PositionsDataset(Dataset):
    """ Dataset for atomic positions."""

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[List[float]],
    ):
        self.positions = [torch.tensor(s.positions).flatten() for s in structures]
        self.labels = [torch.tensor(l).flatten() for l in labels]
        self.num_inputs = len(self.positions[0])

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        x = ()
        x += (self.positions[index],self.labels[index])
        return x
    
    def get_data(self):
        return np.array([d.numpy() for d in self.positions]), np.array([l.numpy() for l in self.labels])

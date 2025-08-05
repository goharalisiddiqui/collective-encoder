import os
import argparse
from typing import List, Dict
from tqdm import tqdm
import random

import numpy as np
import ase

from utils import parse_slice

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

DOUBLE_PRECISION = False

class distancesDataset(Dataset):
    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[float],
        dtype=torch.float32,
        group1 : str = "0",
        group2 : str = "0",
        atm_ids : List[int] =  None
    ):
        assert len(structures) == len(labels), "Number of structures and labels must match"
        group1 = parse_slice(group1)
        group2 = parse_slice(group2)
        
        atns = structures[0].get_atomic_numbers()
        group1_indices = list(range(*group1.indices(len(atns))))
        group2_indices = list(range(*group2.indices(len(atns))))
        
        pairs = []
        for i in group1_indices:
            for j in group2_indices:
                if j > i:
                    pairs.append((i, j))
        self.data_shape = (len(pairs),)
        self.distances = []
        for s in structures:
            distances = []
            positions = s.get_positions()
            for i, j in pairs:
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
            self.distances.append(torch.tensor(distances, dtype=dtype))
        
        self.labels = [torch.tensor(d, dtype=torch.float32) for d in labels]
        self.num_inputs = len(pairs)
        
        if atm_ids is not None:
            print("#########################################")
            print("Atom ID information for distances dataset")
            print("#########################################")
            print(f"ATOM IDs: {atm_ids}")
            for ind, (i, j) in enumerate(pairs):
                print(f"Distance {ind}: {atm_ids[i]} <-> {atm_ids[j]}")
            print("#########################################")
        
    def __len__(self):
        return len(self.distances)

    def __getitem__(self, index):
        x = (self.distances[index],self.labels[index])
        return x
    
    def get_data(self):
        return np.array([d.numpy() for d in self.distances]), np.array([l.numpy() for l in self.labels])

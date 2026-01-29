from typing import List, Optional, Dict
from xml.parsers.expat import model

import numpy as np
import ase

from utils import parse_slice

import torch
from torch.utils.data import Dataset
from torch.nn.functional import pairwise_distance

from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)

from collective_encoder.datasets.base import BaseDataset

class MetatomicDistanceDataset(torch.nn.Module):
    def __init__(self, pairs: List[tuple]):
        super().__init__()
        self.pairs = pairs

        mask_i = []
        mask_j = []
        for i, j in self.pairs:
            mask_i.append(i)
            mask_j.append(j)
        self.register_buffer("mask_i", torch.tensor(mask_i, dtype=torch.long))
        self.register_buffer("mask_j", torch.tensor(mask_j, dtype=torch.long))
    
    def get_atomic_types(self):
        return [a for a in range(0, 119)],  # all elements

    def get_interaction_range(self):
        return torch.inf

    def get_length_unit(self):
        return "nanometer"

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> torch.Tensor:

        pd_batch = torch.stack(
            [pairwise_distance(systems[i].positions.view(-1,3)[self.mask_i], 
                               systems[i].positions.view(-1,3)[self.mask_j]) 
            for i in range(len(systems))], dim=0)
        return pd_batch


class DistancesDataset(Dataset, BaseDataset):
    ''' Dataset for pairwise distances between two groups of atoms.

    The groups can be specified using python slice notation, e.g. "0:3" for the first three atoms.
    The dataset returns the distances between all pairs of atoms in the two groups for each structure.

    Args:
        structures (List[ase.Atoms]): List of ASE Atoms objects representing the structures.
        labels (List[float]): List of labels (e.g. energies) corresponding to each structure.
        group1 (str): Slice notation for the first group of atoms (default: "0").
        group2 (str): Slice notation for the second group of atoms (default: "0").
        atm_ids (List[int], optional): List of atom IDs corresponding to the atoms in the structures. If provided, will print the atom IDs for each distance pair.
    
    Returns:
        distances (torch.Tensor): Tensor of shape (num_structures, num_pairs) containing the distances.
        labels (torch.Tensor): Tensor of shape (num_structures,) containing the labels.
    '''
    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[float],
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
        self.pairs = pairs
        self.distances = []
        for s in structures:
            distances = []
            positions = s.get_positions()
            for i, j in pairs:
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
            self.distances.append(torch.tensor(distances))

        self.labels = [torch.tensor(d) for d in labels]
        self.num_inputs = len(pairs)
        
        if atm_ids is not None:
            self.log_list("Atom IDs", atm_ids)
            for ind, (i, j) in enumerate(pairs):
                self.log_msg(f"Distance {ind}: {atm_ids[i]} <-> {atm_ids[j]}")
        
    def __len__(self):
        return len(self.distances)

    def __getitem__(self, index):
        x = (self.distances[index],self.labels[index])
        return x
    
    def get_data(self):
        return np.array([d.numpy() for d in self.distances]), np.array([l.numpy() for l in self.labels])

    def get_norm_data(self):
        return np.array([d.numpy() for d in self.distances])
    
    def get_metatomic_dataprocessor(self):
        return MetatomicDistanceDataset(self.pairs)
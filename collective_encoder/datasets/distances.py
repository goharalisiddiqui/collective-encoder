from typing import List, Optional, Dict, Union

import numpy as np
import ase

import torch
from torch.utils.data import Dataset
from torch.nn.functional import pairwise_distance
from tqdm import tqdm

from .base import BaseDataset
from gslibs.utils.common import parse_slice


class DistancesDataset(Dataset, BaseDataset):
    """
    Dataset for pairwise distances between two groups of atoms.

    The groups can be specified using python slice notation, e.g. "0:3" for the first three atoms.
    The dataset returns the distances between all pairs of atoms in the two groups for each structure.

    Parameters
    ----------
    structures : list of ase.Atoms
        List of ASE Atoms objects representing the structures.
    labels : list of float
        List of labels (e.g. energies) corresponding to each structure.
    args : dict, optional
        Configuration dictionary containing options like:
        - ``group1`` (str): Slice notation for the first group of atoms.
        - ``group2`` (str): Slice notation for the second group of atoms.
        - ``atm_ids`` (list of int, optional): Atom IDs corresponding to the atoms in the structures.
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "DISTANCES"
    _OPTIONAL_ARGS = {
        'group1': None,
        'group2': None,
        'atm_ids': None,
    }

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[float],
        args: Dict[str, Union[float, int, str]] = None,
        **kwargs,
    ):
        super().__init__(args=args, **kwargs)
        assert len(structures) == len(labels), "Number of structures and labels must match"
        atns = structures[0].get_atomic_numbers()

        if self.group1 is None:
            group1_indices = list(range(len(atns)))
        else:
            group1 = parse_slice(self.group1)
            group1_indices = list(range(*group1.indices(len(atns)))) if group1 != slice(None) else list(range(len(atns)))
        if self.group2 is None:
            group2_indices = list(range(len(atns)))
        else:
            group2 = parse_slice(self.group2)
            group2_indices = list(range(*group2.indices(len(atns)))) if group2 != slice(None) else list(range(len(atns)))
        
        pairs = []
        for i in group1_indices:
            for j in group2_indices:
                if j > i:
                    pairs.append((i, j))
        self.data_shape = (len(pairs),)
        self.pairs = pairs
        self.distances = []
        for s in tqdm(structures, desc="Calculating distances"):
            distances = []
            positions = s.get_positions()
            for i, j in pairs:
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
            self.distances.append(torch.tensor(distances))

        self.labels = [torch.tensor(d) for d in labels]
        self.num_inputs = len(pairs)
        
        if self.atm_ids is not None:
            self.log_info(f"Distance index to atom ID mapping for {len(pairs)} pairs:")
            for ind, (i, j) in enumerate(pairs):
                self.log_msg(f" {ind}: {self.atm_ids[i]} <-> {self.atm_ids[j]}")
        
    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples.
        """
        return len(self.distances)

    def __getitem__(self, index):
        """
        Get the sample at the specified index.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            Tuple of `(distances, labels)` as PyTorch tensors.
        """
        x = (self.distances[index], self.labels[index])
        return x
    
    def get_datapoint_shape(self):
        """
        Return the shape of a single data point's features.

        Returns
        -------
        tuple
            The shape of the extracted features.
        """
        return self.data_shape
    
    def get_data(self):
        """
        Get all distances and labels as NumPy arrays.

        Returns
        -------
        tuple
            Tuple of (distances_array, labels_array).
        """
        return np.array([d.numpy() for d in self.distances]), np.array([l.numpy() for l in self.labels])

    def get_norm_data(self):
        """
        Return array of data used to fit normalizers (scalers).

        Returns
        -------
        numpy.ndarray
            Data to be normalized.
        """
        return np.array([d.numpy() for d in self.distances])
    
    def get_metatomic_dataprocessor(self):
        """
        Get the Metatomic data processor for distance datasets.

        Returns
        -------
        MetatomicDistanceDataset
            A dataset wrapper compatible with Metatomic pipelines.
        """
        return MetatomicDistanceDataset(self.pairs)

# ------------------------------------------------------------------
# Matatomic interface
# ------------------------------------------------------------------

try:
    from metatensor.torch import Labels

    from metatomic.torch import (
        ModelOutput,
        System,
    )
    
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

except ImportError:
    pass
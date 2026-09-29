import os

from typing import Dict, List, Union

import numpy as np
import ase

import torch
from torch.utils.data import Dataset

from .base import BaseDataset


class PositionsDataset(Dataset, BaseDataset):
    """
    Dataset for raw atomic positions.

    Extracts the unflattened coordinate positions from ASE Atoms objects.

    Parameters
    ----------
    structures : list of ase.Atoms
        List of molecular structures.
    labels : list of list of float
        Target labels for each structure.
    dataset_args : dict, optional
        Configuration dictionary (empty for this dataset).
    kwargs : dict
        Additional keyword arguments (e.g. ``tag``).
    """
    
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
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples.
        """
        return len(self.positions)

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
            Tuple of `(positions, labels)` as PyTorch tensors.
        """
        x = ()
        x += (self.positions[index],self.labels[index])
        return x
    
    def get_data(self):
        """
        Get all positions and labels as NumPy arrays.

        Returns
        -------
        tuple
            Tuple of (positions_array, labels_array).
        """
        return np.array([d.numpy() for d in self.positions]), np.array([l.numpy() for l in self.labels])
    
    def get_norm_data(self) -> np.ndarray:
        """
        Return array of data used to fit normalizers (scalers).

        Returns
        -------
        numpy.ndarray
            Data to be normalized.
        """
        return np.array([d.numpy() for d in self.positions])

    def get_datapoint_shape(self) -> tuple:
        """
        Return the shape of a single data point's features.

        Returns
        -------
        tuple
            The shape of the extracted features.
        """
        return tuple(self.positions[0].shape)

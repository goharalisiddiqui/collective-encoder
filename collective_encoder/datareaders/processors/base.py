from abc import ABC
from typing import List, Tuple, Any, Dict, Union
import numpy as np
import torch

from collective_encoder.common.module import CEModule
from collective_encoder.datasets.base import BaseDataset

class BaseDataProcessor(CEModule, ABC):
    """
    Abstract base class for sequence processors.

    Processors modify the behavior of trajectory reading either by
    expanding/altering the indices before reading (via prepare_seq)
    or transforming the trajectory frames and labels after reading
    (via postprocess_seq).
    """
    _IDENTIFIER = ""
    _REQUIRED_ARGS = []
    _OPTIONAL_ARGS = {}
    
    def get_dataset_wrapper(self, 
        dataset_cls: BaseDataset, 
        dataset_args: Dict[str, Any]
        ) -> BaseDataset:
        """
        Return a dataset wrapper class for the processed data.
        By default, it returns a simple wrapper that inherits from the provided dataset class.
        """
        class WrapperDataset(dataset_cls):
            pass

        return WrapperDataset

    def prepare_seq(self, seq: List[int]) -> List[int]:
        """
        Transform a sequence of indices before passing to the parallel worker.

        Parameters
        ----------
        seq : list of int
            Original indices.

        Returns
        -------
        list of int
            Transformed indices.
        """
        return seq

    def postprocess_seq(self, 
        targets: Union[List[Any], np.ndarray, torch.Tensor], 
        labels: Union[List[Any], np.ndarray, torch.Tensor]
        ) -> Tuple[Union[List[Any], np.ndarray, torch.Tensor], Union[List[Any], np.ndarray, torch.Tensor]]:
        """
        Post-process a single sequence's (mol_traj, labels) after reading.

        Parameters
        ----------
        mol_traj : list of ase.Atoms
            The read trajectory frames.
        labels : list of list of float
            The computed labels.

        Returns
        -------
        tuple
            Processed `(mol_traj, labels)`.
        """
        return targets, labels

    def adjust_total_datapoints(self, total: int) -> int:
        """
        Adjust the total number of frames based on the processor's requirements.
        By default, it doesn't change anything.
        """
        return total

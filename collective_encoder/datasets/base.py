from abc import ABC, abstractmethod
from typing import Any, List, Union, Dict, Tuple

import numpy as np
import torch

from collective_encoder import TENSOR_PRECISION
from collective_encoder.common.module import CEModule
from collective_encoder.datareaders.base import BaseDataReader
from torch.utils.data import DataLoader

class BaseDataset(CEModule, ABC):
    """
    Abstract base dataset class for various dataset types.

    This class dictates the interface for creating datasets 
    that act as feature extractors for neural networks.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options.
    kwargs : dict
        Additional keyword arguments forwarded to the parent `CEModule`.
    """
    _IDENTIFIER = "BaseDataset"
    
    @classmethod
    def _prepare_dataset(cls, 
                         dataset_args: Dict[str, Union[float, int, str]], 
                         datareader: BaseDataReader) \
        -> Tuple[type, Dict[str, Union[float, int, str]], type]:
        """
        Prepare the dataset. This method is a placeholder for any preprocessing steps.
        It can be used to inject additional information from the datareader into the dataset arguments.

        Parameters
        ----------
        dataset_args : dict
            Configuration dictionary (not used in this dataset).
        datareader : BaseDataReader
            Data reader instance (to inject information into dataset args).
        """
        return cls, dataset_args, DataLoader
        

    def __init__(self,
                 args: Dict[str, Union[float, int, str]] = None,
                 **kwargs,
                 ):
        super().__init__(args, **kwargs)
        self.dtype = TENSOR_PRECISION

    def _normalize_labels_to_list(
        self,
        labels: Any,
        n_samples: int,
        flatten: bool = False,
    ) -> List[torch.Tensor]:
        """
        Convert ``labels`` (dictionary of tensors/arrays or sequence of arrays) into a
        per-sample list of :class:`torch.Tensor` objects of length ``n_samples``.

        Parameters
        ----------
        labels : dict or list or numpy.ndarray or torch.Tensor
            Input labels from a datareader or caller.
        n_samples : int
            Expected number of samples (first dimension).
        flatten : bool, optional
            Whether to flatten each per-sample label tensor to 1-D.

        Returns
        -------
        list of torch.Tensor
            Per-sample label tensors of length ``n_samples``.
        """
        if isinstance(labels, dict):
            if len(labels) == 0:
                return [torch.empty((0,), dtype=self.dtype) for _ in range(n_samples)]
            cols = []
            for k, v in labels.items():
                t = torch.as_tensor(v, dtype=self.dtype)
                if t.ndim == 1:
                    t = t.unsqueeze(1)
                elif t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)
                if t.shape[0] != n_samples:
                    self.raise_error(
                        f"Number of structures ({n_samples}) and labels for key '{k}' ({t.shape[0]}) must match"
                    )
                cols.append(t)
            combined = torch.cat(cols, dim=1) if len(cols) > 1 else cols[0]
            return [
                combined[i].flatten() if flatten else combined[i]
                for i in range(n_samples)
            ]

        assert len(labels) == n_samples, "Number of structures and labels must match"
        return [
            torch.as_tensor(d, dtype=self.dtype).flatten()
            if flatten
            else torch.as_tensor(d, dtype=self.dtype)
            for d in labels
        ]
    
    @abstractmethod
    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples.
        """
        raise NotImplementedError("Subclasses must implement __len__ method")
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Get the sample at the specified index.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        Any
            The processed data sample (often a tuple of features and labels).
        """
        raise NotImplementedError("Subclasses must implement __getitem__ method")
    
    @abstractmethod
    def get_norm_data(self) -> np.ndarray:
        """
        Return array of data used to fit normalizers (scalers).

        Returns
        -------
        numpy.ndarray
            Data to be normalized.
        """
        raise NotImplementedError("Subclasses must implement get_norm_data method")
    
    @abstractmethod
    def get_datapoint_shape(self) -> tuple:
        """
        Return the shape of a single data point's features.

        Returns
        -------
        tuple
            The shape of the extracted features.
        """
        raise NotImplementedError("Subclasses must implement get_datapoint_shape method")
    
    @abstractmethod
    def get_data(self):
        """
        Return whole dataset (features and labels).

        Returns
        -------
        tuple
            Tuple of (distances_array, labels_array).
        """
        raise NotImplementedError("Subclasses must implement get_data method")

    def get_labels(self) -> Any:
        """
        Return the labels for the dataset.
        
        Returns
        -------
        Any
            The labels for the dataset.
        """
        return self.get_data()[1]
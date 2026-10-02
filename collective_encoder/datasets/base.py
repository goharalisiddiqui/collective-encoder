from abc import ABC, abstractmethod
from typing import Any, List, Union, Dict

import numpy as np

from collective_encoder.common.module import CEModule
from collective_encoder.common.config_check import validate_required_fields

class BaseDataset(CEModule, ABC):
    """
    Abstract base dataset class for various dataset types.

    This class inherits from `CEModule` and dictates the interface for
    creating datasets that act as feature extractors for neural networks.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options.
    kwargs : dict
        Additional keyword arguments forwarded to the parent `CEModule`.
    """

    def __init__(self,
                 args: Dict[str, Union[float, int, str]] = None,
                 **kwargs,
                 ):
        super().__init__(args, **kwargs)
    
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
    
    @classmethod
    def get_identifier(cls) -> str:
        """
        Get the string identifier for this dataset type.

        Returns
        -------
        str
            The identifier string.
        """
        if cls._IDENTIFIER is None:
            raise NotImplementedError(f"{cls.__name__} must define a class-level _IDENTIFIER attribute")
        return cls._IDENTIFIER
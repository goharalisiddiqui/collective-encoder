from abc import ABC, abstractmethod
from typing import Any, List, Union, Dict

import numpy as np

from collective_encoder.common.module import CEModule
from collective_encoder.common.config_check import validate_required_fields

class BaseDataset(CEModule, ABC):
    ''' Abstract base dataset class for various dataset types.
    '''
    _IDENTIFIER: str = None             # A unique identifier for the dataset type, used for registry and config validation
    _REQUIRED_ARGS: List[str] = []      # List of required keys in dataset_args for this dataset type, used for config validation
    _OPTIONAL_ARGS: Dict[str, Union[float, int, str]] = {}      # Dict of optional keys and their default values in dataset_args for this dataset type, used for config validation
    
    def __init__(self,
                 dataset_args: Dict[str, Union[float, int, str]] = None,
                 **kwargs,
                 ):
        validate_required_fields(dataset_args, fields=self._REQUIRED_ARGS)
        for key, default_value in self._OPTIONAL_ARGS.items():
            if key not in dataset_args:
                dataset_args[key] = default_value
        for key in dataset_args:
            self.__setattr__(key, dataset_args[key]) # Set dataset_args as attributes of the dataset for easy access
        super().__init__(**kwargs)
    
    @abstractmethod
    def __len__(self):
        ''' Return the number of samples in the dataset.
        '''
        raise NotImplementedError("Subclasses must implement __len__ method")
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        ''' Get the sample at the specified index.
        '''
        raise NotImplementedError("Subclasses must implement __getitem__ method")
    
    @abstractmethod
    def get_norm_data(self) -> np.ndarray:
        ''' Return array of data to be normalized.
        '''
        raise NotImplementedError("Subclasses must implement get_norm_data method")
    
    @abstractmethod
    def get_datapoint_shape(self) -> tuple:
        ''' Return the shape of a single data point.
        '''
        raise NotImplementedError("Subclasses must implement get_datapoint_shape method")
    
    @classmethod
    def get_identifier(cls) -> str:
        '''
        Get the identifier for this DataReader type.

        Returns:
            str: The identifier string.
        '''
        if cls._IDENTIFIER is None:
            raise NotImplementedError(f"{cls.__name__} must define a class-level _IDENTIFIER attribute")
        return cls._IDENTIFIER
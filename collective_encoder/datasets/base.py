from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from collective_encoder.common.module import CEModule

class BaseDataset(CEModule, ABC):
    ''' Abstract base dataset class for various dataset types.
    '''
    def __init__(self,
                 **kwargs,
                 ):
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
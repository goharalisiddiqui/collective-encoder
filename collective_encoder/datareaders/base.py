from abc import ABC, abstractmethod
from typing import List

from collective_encoder.common.module import CEModule

class BaseDataReader(CEModule, ABC):
    '''
    Abstract base class for trajectory readers.
    '''
    # To be overridden by subclasses
    _IDENTIFIER: str = None
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def get_total_frames(self):
        '''
        Method to get the total number of frames in the trajectory.
        '''
        raise NotImplementedError("Subclasses must implement get_total_frames method")
    
    def get_label_names(self) -> List[str]:
        '''
        Get the names of the collective variables (labels).

        Returns:
            List[str]: List of column names
        '''
        
        if not hasattr(self, 'label_list'):
            raise AttributeError(f"{type(self).__name__} does not have "
                                 "'label_list' attribute. Ensure that the "
                                 "trajectory has been read and labels have been "
                                 "computed before calling get_label_names.")
        return self.label_list

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
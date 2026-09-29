from abc import ABC, abstractmethod
from typing import List

from collective_encoder.common.module import CEModule

class BaseDataReader(CEModule, ABC):
    """
    Abstract base class for data readers.
    """
    
    @abstractmethod
    def get_total_frames(self):
        """
        Method to get the total number of frames in the trajectory.

        Returns
        -------
        int
            The total number of frames or records available.
        """
        raise NotImplementedError("Subclasses must implement get_total_frames method")

    @classmethod
    def get_identifier(cls) -> str:
        """
        Get the identifier for this DataReader type.

        Returns
        -------
        str
            The identifier string.
        """
        if cls._IDENTIFIER is None:
            raise NotImplementedError(f"{cls.__name__} must define a class-level _IDENTIFIER attribute")
        return cls._IDENTIFIER
    
    def get_label_names(self) -> List[str]:
        """
        Get the names of the collective variables (labels).

        Returns
        -------
        list of str
            List of column names for the labels.

        Raises
        ------
        AttributeError
            If `label_list` has not been set (usually done during reading).
        """
        
        if not hasattr(self, 'label_list'):
            raise AttributeError(f"{type(self).__name__} does not have "
                                 "'label_list' attribute. Ensure that the "
                                 "trajectory has been read and labels have been "
                                 "computed before calling get_label_names.")
        return self.label_list
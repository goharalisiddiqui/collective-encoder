from abc import ABC, abstractmethod
from typing import List

from collective_encoder.common.module import CEModule
from collective_encoder.datalabelers.base import BaseLabeler

class BaseDataReader(CEModule, ABC):
    """
    Abstract base class for data readers.
    """
    _COMPATIBLE_LABELERS = ["CONCAT", "DUMMY"]

    
    @abstractmethod
    def read(self,
            labeler_type: str,
            labeler_args: dict,
            **kwargs):
        """
        Abstract method to read the trajectory data and compute labels.

        Parameters
        ----------
        labeler_type : str
            The type of labeler to use for computing labels.
        labeler_args : dict
            Arguments to pass to the labeler for label computation.
        **kwargs
            Additional keyword arguments for reading the trajectory.
        """
        raise NotImplementedError("Subclasses must implement read method")

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
    
    def initialize_labeler(self, labeler_type: str, labeler_args: dict) -> BaseLabeler:
        """
        Initialize the labeler based on the provided type and arguments.

        Parameters
        ----------
        labeler_type : str
            The type of labeler to use for computing labels.
        labeler_args : dict
            Arguments to pass to the labeler for label computation.
        """
        if labeler_type is None:
            labeler_type = "DUMMY" 
        if labeler_type not in self._COMPATIBLE_LABELERS:
            raise ValueError(f"Labeler type '{labeler_type}' is not compatible with this data reader. "
                             f"Compatible labelers: {self._COMPATIBLE_LABELERS}")
        
        return self._initialize_labeler(labeler_type, labeler_args)
    
    @abstractmethod
    def _initialize_labeler(self, labeler_type: str, labeler_args: dict) -> BaseLabeler:
        """
        Internal method to initialize the labeler. Subclasses should implement this.

        Parameters
        ----------
        labeler_type : str
            The type of labeler to use for computing labels.
        labeler_args : dict
            Arguments to pass to the labeler for label computation.
        """
        raise NotImplementedError("Subclasses must implement _initialize_labeler method")
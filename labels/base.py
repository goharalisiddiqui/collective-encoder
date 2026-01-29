from abc import ABC
import MDAnalysis as mda
from typing import Dict, List, Union

class BaseLabeler(ABC):
    '''
    Abstract base class for labelers.
    '''

    def get_names(self) -> List[str]:
        """ Get the names of the labels. """
        raise NotImplementedError("Subclasses must implement get_names method")

    def compute(self) -> List[float]:
        """ Compute the labels for the current frame. """
        raise NotImplementedError("Subclasses must implement compute method")
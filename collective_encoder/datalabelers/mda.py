from abc import ABC
import numpy as np
import MDAnalysis as mda
from typing import Optional, Dict, List, Union

from .base import BaseLabeler


class MDALabeler(BaseLabeler, ABC):
    """
    Base class for MDAnalysis-based labelers.
    
    
    """

    def reinitialize(self, universe: mda.Universe) -> None:
        self.__init__(universe=universe, args=self.get_args(), **self.get_run_args())

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union

import pandas as pd
import numpy as np
import MDAnalysis as mda

from collective_encoder.common.module import CEModule


class BaseLabeler(CEModule, ABC):
    """
    Abstract base class for all labelers.

    Concrete labelers must subclass either :class:`FrameLabeler` (for
    MDAnalysis universe-based, per-frame computation) or
    :class:`BatchLabeler` (for DataFrame index-based batch computation).

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options.
    kwargs : dict
        Additional keyword arguments forwarded to the parent `CEModule`.
    """
    def reinitialize(self, **args) -> None:
        """
        Reinitialize the labeler with new arguments.

        This method is intended to be overridden by subclasses that need to
        reinitialize their state when the underlying data (e.g., MDAnalysis
        Universe or DataFrame) changes. The default implementation simply
        calls the constructor with the new arguments.

        Parameters
        ----------
        *args : tuple
            Positional arguments for reinitialization.
        """
        pass

    @abstractmethod
    def get_label_names(self) -> List[str]:
        """
        Return the names of the label columns produced by this labeler.

        Returns
        -------
        list of str
            Names of the generated label columns.
        """
        raise NotImplementedError
    
    @abstractmethod
    def compute(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute labels for the given DataFrame row indices.

        Parameters
        ----------
        indices : list of int
            Integer row indices into the DataFrame supplied at construction time.

        Returns
        -------
        numpy.ndarray
            NumPy array of shape ``(len(indices), n_labels)``.
        """
        raise NotImplementedError

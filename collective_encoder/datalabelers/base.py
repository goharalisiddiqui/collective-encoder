from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np

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

    def __init__(self,
                 args: Dict[str, Union[float, int, str]] = None,
                 **kwargs):
        if args is None:
            args = {}
        super().__init__(args=args, **kwargs)

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


class FrameLabeler(BaseLabeler):
    """Abstract base for universe-based labelers (XTC / MDAnalysis).

    The labeler is initialised with a positioned MDAnalysis ``Universe``
    and ``compute()`` is called once per trajectory frame.  The current
    frame is implicit in the universe; no index argument is needed.
    """

    @abstractmethod
    def compute(self) -> List[float]:
        """
        Compute labels for the current trajectory frame.

        Returns
        -------
        list of float
            List of scalar label values, one per label column.
        """
        raise NotImplementedError


class BatchLabeler(BaseLabeler):
    """Abstract base for DataFrame index-based labelers (PLUMED / tabular).

    The labeler is initialised with a ``pd.DataFrame`` and ``compute()``
    is called with a list of integer row indices to label in batch.
    """

    @abstractmethod
    def compute(self, indices: List[int]) -> np.ndarray:
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
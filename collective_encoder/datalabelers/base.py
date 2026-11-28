from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from collective_encoder.common.module import CEModule


class BaseLabeler(CEModule, ABC):
    """
    Abstract base class for all labelers.

    Every labeler accepts an optional ``name`` parameter in ``args`` and returns
    a dictionary mapping label names to NumPy arrays from :meth:`compute`.
    For labelers that output a single feature vector, the dictionary key defaults
    to the lowercase class name (``self.__class__.__name__.lower()``) when
    ``name`` is not provided.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options. Optional keys include:
        - ``name`` (str, optional): Custom name for the output label key.
    kwargs : dict
        Additional keyword arguments forwarded to the parent :class:`CEModule`.
    """

    _OPTIONAL_ARGS: Dict[str, Any] = {
        "name": None,
    }

    def get_default_label_name(self) -> str:
        """
        Return the configured label name ``self.name`` if provided, otherwise
        the lowercase class name.

        Returns
        -------
        str
            Resolved default label key name.
        """
        if getattr(self, "name", None) is not None and str(self.name).strip():
            return str(self.name).strip()
        return self.__class__.__name__.lower()

    def reinitialize(self, **args) -> None:
        """
        Reinitialize the labeler with new arguments.

        Parameters
        ----------
        **args : dict
            Keyword arguments for reinitialization.
        """
        pass

    @abstractmethod
    def get_label_names(self) -> List[str]:
        """
        Return the names of the label keys produced by this labeler.

        Returns
        -------
        list of str
            Names of the generated label dictionary keys.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Compute labels for the current frame or given row indices.

        Parameters
        ----------
        indices : list of int, optional
            Integer row/frame indices.

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary mapping each label name to its computed NumPy array.
        """
        raise NotImplementedError

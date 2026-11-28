import numpy as np
from typing import Dict, List, Optional, Union

from .base import BaseLabeler


class DummyLabeler(BaseLabeler):
    """
    No-op labeler for unsupervised learning pipelines.

    Compatible with both :class:`FrameLabeler` (XTC/MDAnalysis) and
    :class:`BatchLabeler` (PLUMED/DataFrame) call sites:

    - Called as ``compute()`` → returns ``[0.0]`` (one zero per frame).
    - Called as ``compute(indices)`` → returns a zeros array of shape
      ``(len(indices), 1)``.

    Parameters
    ----------
    args : dict, optional
        Ignored. Present only for interface compatibility.
    kwargs : dict
        Forwarded to parent (universe/dataframe are silently ignored).
    """

    _IDENTIFIER = "DUMMY"
    _REQUIRED_ARGS = []
    _OPTIONAL_ARGS = BaseLabeler._OPTIONAL_ARGS.copy()

    def __init__(self,
                 args: Dict[str, Union[str, List[str]]] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(args=args, **kwargs)
        pass
    
    def _print_init_banner(self):
        pass # Suppress banner for dummy labeler

    def get_label_names(self) -> List[str]:
        """
        Return the single dummy label name.

        Returns
        -------
        list of str
            List containing the configured ``name`` or ``['dummylabeler']``.
        """
        return [self.get_default_label_name()]

    def compute(self, indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Return zeros as a dictionary; works for both frame-based and batch-based call sites.

        Parameters
        ----------
        indices : list of int, optional
            If provided, returns a 2-D zeros array of shape ``(len(indices), 1)``.

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary mapping the label name to a zeros array.
        """
        key = self.get_default_label_name()
        if indices is None:
            return {key: np.array([0.0], dtype=float)}
        return {key: np.zeros((len(indices), 1), dtype=float)}

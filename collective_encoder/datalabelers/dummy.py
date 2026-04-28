import numpy as np
from typing import Dict, List, Optional, Union

from .base import BaseLabeler


class DummyLabeler(BaseLabeler):
    """No-op labeler for unsupervised learning pipelines.

    Compatible with both :class:`FrameLabeler` (XTC/MDAnalysis) and
    :class:`BatchLabeler` (PLUMED/DataFrame) call sites:

    - Called as ``compute()`` → returns ``[0.0]`` (one zero per frame).
    - Called as ``compute(indices)`` → returns a zeros array of shape
      ``(len(indices), 1)``.

    Args:
        args: Ignored.  Present only for interface compatibility.
        **kwargs: Forwarded to parent (universe/dataframe are silently ignored).
    """

    _IDENTIFIER = "DUMMY"
    _REQUIRED_ARGS = []
    _OPTIONAL_ARGS = {}

    def __init__(self,
                 args: Dict[str, Union[str, List[str]]] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(args=args, **kwargs)
        pass

    def get_label_names(self) -> List[str]:
        return ['dummy']

    def compute(self, indices: Optional[List[int]] = None) -> Union[List[float], np.ndarray]:
        """Return zeros; works for both frame-based and batch-based call sites."""
        if indices is None:
            return [0.0]
        return np.zeros((len(indices), 1))

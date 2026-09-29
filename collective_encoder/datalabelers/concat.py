import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import MDAnalysis as mda
import numpy as np
import pandas as pd

from .base import BaseLabeler
from .resolver import get_labeler

_log = logging.getLogger(__name__)


class ConcatLabeler(BaseLabeler):
    """Concatenate labels from multiple sub-labelers in sequential order.

    Accepts a list of sub-labeler definitions in ``labelers`` and executes them
    sequentially, concatenating their label names and computed feature values.
    Supports both universe-based FrameLabelers (XTC) and DataFrame-based BatchLabelers (PLUMED).

    Args:
        universe (mda.Universe, optional): MDAnalysis Universe for universe-based FrameLabelers.
        dataframe (pd.DataFrame, optional): pandas DataFrame for DataFrame-based BatchLabelers.
        args: Configuration dict with key ``labelers`` (required) — a list of child labeler
            definitions. Each entry can be:
            - Dict: ``{"labeler_type": "...", "labeler_args": {...}}`` (or ``{"type": "...", "args": {...}}``)
            - Tuple/List: ``["LABELER_TYPE", labeler_args_dict]``
    """

    _IDENTIFIER = "CONCAT"
    _REQUIRED_ARGS = ["labelers"]
    _OPTIONAL_ARGS = {}

    def __init__(
        self,
        universe: Optional[mda.Universe] = None,
        dataframe: Optional[pd.DataFrame] = None,
        args: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(args=args, **kwargs)

        if not self.labelers or len(self.labelers) == 0:
            self.raise_error("'labelers' list must be provided and non-empty in args")

        self.child_labelers = []

        child_kwargs = {}
        if universe is not None:
            child_kwargs["universe"] = universe
        if dataframe is not None:
            child_kwargs["dataframe"] = dataframe
        child_kwargs.update(kwargs)

        for idx, item in enumerate(self.labelers):
            if isinstance(item, dict):
                child_type = item.get("labeler_type") or item.get("type")
                child_args = item.get("labeler_args") or item.get("args") or {}
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                child_type = item[0]
                child_args = item[1] or {}
            else:
                self.raise_error(f"Invalid labeler specification at index {idx}: {item}")

            if not child_type:
                self.raise_error(f"Missing labeler_type at index {idx} in 'labelers'")

            labeler_cls = get_labeler(child_type)
            child_instance = labeler_cls(args=child_args, **child_kwargs)
            self.child_labelers.append(child_instance)

    def get_label_names(self) -> List[str]:
        names = []
        for child in self.child_labelers:
            names.extend(child.get_label_names())
        return names

    def compute(self, indices: Optional[List[int]] = None) -> Union[List[float], np.ndarray]:
        """Compute concatenated labels for all child labelers.

        Args:
            indices (List[int], optional): DataFrame row indices for BatchLabeler mode.

        Returns:
            Flat list of floats for FrameLabeler mode, or 2D NumPy array for BatchLabeler mode.
        """
        if indices is not None:
            batch_results = []
            for child in self.child_labelers:
                res = child.compute(indices)
                if not isinstance(res, np.ndarray):
                    res = np.array(res)
                if res.ndim == 1:
                    res = res.reshape(-1, 1)
                batch_results.append(res)
            return np.hstack(batch_results)
        else:
            flat_results = []
            for child in self.child_labelers:
                res = child.compute()
                if isinstance(res, np.ndarray):
                    flat_results.extend(res.flatten().tolist())
                elif isinstance(res, (list, tuple)):
                    flat_results.extend(res)
                else:
                    flat_results.append(float(res))
            return flat_results

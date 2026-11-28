from typing import Any, Dict, List, Optional

import MDAnalysis as mda
import numpy as np
import pandas as pd

from collective_encoder.common.registry import get_module
from .base import BaseLabeler


class ConcatLabeler(BaseLabeler):
    """
    Concatenate and merge dictionary labels from multiple child labelers in order.

    Executes each child labeler sequentially. For each ``(key, val)`` in a child's
    returned dictionary:
    - If ``key`` already exists in the accumulated output dictionary, concatenates
      ``val`` onto the existing array along the feature dimension.
    - Otherwise, inserts ``key`` as a new entry in the output dictionary.

    Parameters
    ----------
    universe : MDAnalysis.Universe, optional
        MDAnalysis Universe for universe-based labelers.
    dataframe : pandas.DataFrame, optional
        pandas DataFrame for DataFrame-based labelers.
    args : dict
        Configuration dictionary with key ``labelers`` (required) — a list of child
        labeler definitions.
    kwargs : dict
        Additional keyword arguments forwarded to child labelers and :class:`BaseLabeler`.
    """

    _IDENTIFIER = "CONCAT"
    _REQUIRED_ARGS = ["labelers"]
    _OPTIONAL_ARGS = BaseLabeler._OPTIONAL_ARGS.copy()

    def __init__(
        self,
        universe: Optional[mda.Universe] = None,
        dataframe: Optional[pd.DataFrame] = None,
        args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(args=args, **kwargs)

        if not self.labelers or len(self.labelers) == 0:
            self.raise_error("'labelers' list must be provided and non-empty in args")

        self.child_labelers: List[BaseLabeler] = []

        child_kwargs: Dict[str, Any] = {}
        if universe is not None:
            child_kwargs["universe"] = universe
        if dataframe is not None:
            child_kwargs["dataframe"] = dataframe
        child_kwargs.update(kwargs)

        for idx, item in enumerate(self.labelers):
            if isinstance(item, dict):
                child_type = item.get("labeler_type") or item.get("type")
                child_args = dict(item.get("labeler_args") or item.get("args") or {})
                if "name" in item and "name" not in child_args:
                    child_args["name"] = item["name"]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                child_type = item[0]
                child_args = dict(item[1] or {})
            else:
                self.raise_error(f"Invalid labeler specification at index {idx}: {item}")

            if not child_type:
                self.raise_error(f"Missing labeler_type at index {idx} in 'labelers'")

            labeler_cls = get_module("datalabelers", child_type)
            child_instance = labeler_cls(args=child_args, **child_kwargs)
            self.child_labelers.append(child_instance)

    def reinitialize(self, *args, **kwargs) -> None:
        """
        Reinitialize all child labelers with the provided arguments.
        """
        for child in self.child_labelers:
            child.reinitialize(*args, **kwargs)

    def get_label_names(self) -> List[str]:
        """
        Return the ordered list of unique label dictionary keys across all child labelers.

        Returns
        -------
        list of str
            Unique label keys produced by this labeler.
        """
        names: List[str] = []
        for child in self.child_labelers:
            for name in child.get_label_names():
                if name not in names:
                    names.append(name)
        return names

    def compute(self, indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Compute and merge dictionary labels from all child labelers.

        Parameters
        ----------
        indices : list of int, optional
            Row/frame indices forwarded to each child labeler.

        Returns
        -------
        dict of str to numpy.ndarray
            Merged dictionary of label arrays.
        """
        merged: Dict[str, np.ndarray] = {}
        for child in self.child_labelers:
            child_out = child.compute(indices)
            if not isinstance(child_out, dict):
                default_key = child.get_default_label_name()
                child_out = {default_key: np.asarray(child_out, dtype=float)}

            for key, val in child_out.items():
                val_arr = np.asarray(val, dtype=float)
                if key in merged:
                    existing = merged[key]
                    if existing.ndim == 1 and val_arr.ndim == 1:
                        merged[key] = np.concatenate([existing, val_arr], axis=0)
                    else:
                        if existing.ndim == 1:
                            existing = existing.reshape(-1, 1)
                        if val_arr.ndim == 1:
                            val_arr = val_arr.reshape(-1, 1)
                        merged[key] = np.concatenate([existing, val_arr], axis=-1)
                else:
                    merged[key] = val_arr

        return merged

import itertools
from typing import Any, Dict, List, Optional, Union
import numpy as np
import MDAnalysis as mda

from .base import BaseLabeler


class FromExternalLabeler(BaseLabeler):
    """
    Extract pre-computed labels from an external dataset for each trajectory frame.

    Selects one or more named arrays from an external dataset (e.g. loaded
    from an ``.npz`` file or dictionary of arrays) and extracts the values
    for the current frame (using the MDAnalysis ``Universe`` position).

    Multi-dimensional arrays (e.g. forces of shape ``(N, n_atoms, 3)``)
    are flattened along their trailing dimensions, with corresponding
    multi-index label names generated (e.g. ``F_0_0``, ``F_0_1``, etc.).

    Parameters
    ----------
    dataset : dict or mapping, optional
        Mapping of key names to NumPy arrays (e.g., ``raw_data`` from
        :class:`~collective_encoder.datareaders.external_dataset.ExternalDatasetReaderBase`).
        Can also be passed via ``kwargs`` as ``dataset`` or ``raw_data``.
    universe : MDAnalysis.Universe, optional
        Positioned MDAnalysis Universe representing the trajectory frames.
    args : dict
        Configuration dict with key ``keys`` (required) — a string or list
        of string keys to extract from the dataset (e.g. ``['E', 'F']``).
    kwargs : dict
        Additional keyword arguments forwarded to the parent class.

    Raises
    ------
    ValueError
        If ``dataset`` is not provided, if ``keys`` is missing or empty,
        or if any requested key is not present in ``dataset``.
    """

    _IDENTIFIER = "FROM_EXTERNAL"
    _REQUIRED_ARGS = ["keys"]
    _OPTIONAL_ARGS = {}

    def __init__(
        self,
        dataset: Optional[Union[Dict[str, Any], Any]],
        args: Dict[str, Union[str, List[str]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(args=args, **kwargs)

        self.dataset = dataset

        if isinstance(self.keys, str):
            self.keys = [self.keys]
        elif not isinstance(self.keys, list):
            self.raise_error("'keys' must be a list of strings or a single string")

        if len(self.keys) == 0:
            self.raise_error("'keys' must be provided and non-empty in args")

        available_keys = set(self.dataset.keys()) if hasattr(self.dataset, "keys") else set(dir(self.dataset))
        missing_keys = [k for k in self.keys if k not in available_keys]
        if missing_keys:
            self.raise_error(
                f"The following specified keys are not present in the dataset: {missing_keys}. "
                f"Available keys: {sorted(list(available_keys))}"
            )

        self.label_names = self._build_label_names()

    def _build_label_names(self) -> List[str]:
        """
        Generate column names for all selected keys, accounting for array shapes.
        """
        names = []
        for k in self.keys:
            arr = np.asarray(self.dataset[k])
            if arr.ndim == 1:
                names.append(k)
            elif arr.ndim == 2 and arr.shape[1] == 1:
                names.append(k)
            else:
                trailing_dims = arr.shape[1:]
                for multi_idx in itertools.product(*[range(d) for d in trailing_dims]):
                    idx_str = "_".join(str(i) for i in multi_idx)
                    names.append(f"{k}_{idx_str}")
        return names

    def get_label_names(self) -> List[str]:
        """
        Return the names of the label columns.

        Returns
        -------
        list of str
            Names of the generated label columns.
        """
        return self.label_names

    def compute(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute labels for the current frame (FrameLabeler) or given indices (batch).

        Parameters
        ----------
        indices : int, list of int, or numpy.ndarray, optional
            If provided, extracts labels for the given batch of row indices as a 2D NumPy array.
            If None, extracts labels for the current trajectory frame from ``self.universe``.

        Returns
        -------
        list of float or numpy.ndarray
            List of floats if called without indices (frame mode),
            or 2D NumPy array if called with indices (batch mode).
        """
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        n_samples = len(indices)
        blocks = []
        for k in self.keys:
            arr = np.asarray(self.dataset[k])
            sub = arr[indices]
            if sub.ndim == 1:
                sub = sub.reshape(n_samples, 1)
            elif sub.ndim > 2:
                sub = sub.reshape(n_samples, -1)
            elif sub.ndim == 2:
                pass
            blocks.append(sub)

        if not blocks:
            return np.empty((n_samples, 0))

        return np.hstack(blocks)

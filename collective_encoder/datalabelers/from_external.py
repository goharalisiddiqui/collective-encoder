import itertools
from typing import Any, Dict, List, Optional, Union
import numpy as np

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
    _OPTIONAL_ARGS = BaseLabeler._OPTIONAL_ARGS.copy()

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
        Return the dictionary key name for this labeler.

        Returns
        -------
        list of str
            Single-element list containing ``self.get_default_label_name()``.
        """
        return [self.get_default_label_name()]

    def compute(self, indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Compute labels for the given indices and return a dictionary.

        Parameters
        ----------
        indices : int, list of int, or numpy.ndarray, optional
            Frame or batch row indices to extract from ``self.dataset``.

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary mapping ``self.get_default_label_name()`` to a 2-D NumPy
            array of extracted values.
        """
        key = self.get_default_label_name()
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
            return {key: np.empty((n_samples, 0), dtype=float)}

        return {key: np.asarray(np.hstack(blocks), dtype=float)}

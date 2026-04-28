from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseDataset


class ColvarDataset(Dataset, BaseDataset):
    """Dataset for collective variable data from PLUMED output files.

    Each data point is a 1-D feature vector produced by
    :class:`~collective_encoder.datareaders.plumed_output.PlumedOutputReader`.
    Unlike coordinate datasets this class accepts a 2-D numpy array
    ``(n_frames, n_features)`` rather than a list of ASE :class:`ase.Atoms`
    objects — the two formats share the same constructor keyword so that
    :class:`~collective_encoder.datamodules.coordinates.CoordinatesDataModule`'s
    ``_create_datasets()`` method works unchanged.
    """

    _IDENTIFIER = "COLVAR"
    _REQUIRED_ARGS: List[str] = []
    _OPTIONAL_ARGS: Dict[str, object] = {}

    def __init__(
        self,
        structures: Union[np.ndarray, List[np.ndarray]],
        labels: Union[np.ndarray, List[np.ndarray]],
        dataset_args: Dict[str, Union[float, int, str]] = None,
        **kwargs,
    ):
        Dataset.__init__(self)
        BaseDataset.__init__(self, dataset_args=dataset_args, **kwargs)

        # structures: 2-D array (n_frames, n_features) or list of 1-D arrays
        if isinstance(structures, np.ndarray) and structures.ndim == 2:
            self.data = [
                torch.tensor(structures[i], dtype=torch.float32)
                for i in range(len(structures))
            ]
        else:
            self.data = [torch.tensor(s, dtype=torch.float32) for s in structures]

        self.labels = [torch.tensor(l, dtype=torch.float32).flatten() for l in labels]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def get_norm_data(self) -> np.ndarray:
        return np.array([d.numpy() for d in self.data])

    def get_datapoint_shape(self) -> tuple:
        return tuple(self.data[0].shape)

from typing import List, Dict, Any, Union, Tuple
import einops
import numpy as np
import torch
from collective_encoder.datareaders.processors.base import BaseDataProcessor
from collective_encoder.datasets.base import BaseDataset


def _chunk(x, chunk_length, name):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        try:
            return einops.rearrange(x, '... (a b) d -> ... a b d', b=chunk_length)
        except einops.EinopsError as e:
            raise ValueError(
                f"Error rearranging {name} with shape {tuple(x.shape)} "
                f"and chunk_length {chunk_length}: {e}"
            ) from e
    elif isinstance(x, list):   
        return [x[i:i + chunk_length] for i in range(0, len(x), chunk_length)]
    elif isinstance(x, dict):
        return {k: _chunk(v, chunk_length, f"{name}[{k}]") for k, v in x.items()}
    else:
        raise TypeError(
            f"Unsupported type for {name}: {type(x)}. "
            f"Expected np.ndarray, torch.Tensor, or list."
        )

class ChunkMixin:
    _base_cls = None
    chunk_length = None

    def __getitem__(self, index):
        targets, labels = super().__getitem__(index)
        targets = _chunk(targets, self.chunk_length, "targets")
        return targets, labels

    def __reduce__(self):
        # (callable, args, state)
        return (_rebuild, (self._base_cls, self.chunk_length), self.__dict__)

def make_chunked_cls(base_cls, chunk_length):
    return type(
        f"Chunked{base_cls.__name__}",
        (ChunkMixin, base_cls),
        {"_base_cls": base_cls, "chunk_length": chunk_length},
    )

def _rebuild(base_cls, chunk_length):
    cls = make_chunked_cls(base_cls, chunk_length)
    return cls.__new__(cls)

class ChunkProcessor(BaseDataProcessor):
    """
    Turn a sequence of start indices into consecutive frame ranges of length `chunk_length`.
    """
    
    _IDENTIFIER = "CHUNK"
    _REQUIRED_ARGS = [
        'chunk_length',
    ]
    _COMPATIBLE_DATASETS = ["TIMESERIES", "ATOMISTIC_DARTS"]
    
    def get_dataset_wrapper(self, 
        dataset_cls: BaseDataset, 
        dataset_args: Dict[str, Any]
        ) -> BaseDataset:
        """
        Return a dataset wrapper class for the processed data.
        By default, it returns a simple wrapper that inherits from the provided dataset class.
        """
        if dataset_cls._IDENTIFIER not in self._COMPATIBLE_DATASETS:
            self.raise_error(
                f"{self._IDENTIFIER} Processor is not compatible with dataset {dataset_cls._IDENTIFIER}. "
                f"Compatible datasets: {self._COMPATIBLE_DATASETS}"
            )
        return make_chunked_cls(dataset_cls, self.chunk_length)

    def prepare_seq(self, seq: List[int]) -> List[int]:
        """
        Expand start indices into consecutive frame ranges of length sequence_length.
        """
        return [j for i in seq for j in range(i, i + self.chunk_length)]

from typing import List, Dict, Any, Union, Tuple
import einops
import numpy as np
import torch
from collective_encoder.datareaders.processors.base import BaseDataProcessor
from collective_encoder.datasets.base import BaseDataset

class ChunkProcessor(BaseDataProcessor):
    """
    Turn a sequence of start indices into consecutive frame ranges of length `chunk_length`.
    """
    
    _IDENTIFIER = "CHUNK"
    _REQUIRED_ARGS = [
        'chunk_length',
    ]
    _COMPATIBLE_DATASETS = ["TIMESERIES"]
    
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
        chunk_length = self.chunk_length

        class WrapperDataset(dataset_cls):
            
            def __getitem__(self, index: int) -> Any:
                targets, labels = super().__getitem__(index)
                if isinstance(targets, (np.ndarray, torch.Tensor)):
                    try:
                        targets = einops.rearrange(targets, 
                                    '... (a b) d -> ... a b d', 
                                    b=chunk_length
                                )
                    except Exception as e:
                        self.raise_error(f"Error in rearranging targets with "
                                        f"shape {targets.shape} and chunk_length "
                                        f"{chunk_length}: {e}")
                else:
                    targets = [targets[i:i + chunk_length] \
                        for i in range(0, len(targets), chunk_length)]

                # if isinstance(labels, (np.ndarray, torch.Tensor)):
                #     try:
                #         labels = einops.rearrange(labels, 
                #                     '... (a b) d -> ... a b d', 
                #                     b=chunk_length
                #                 )
                #     except Exception as e:
                #         self.raise_error(f"Error in rearranging labels with "
                #                         f"shape {labels.shape} and chunk_length "
                #                         f"{chunk_length}: {e}")
                # else:
                #     labels = [labels[i:i + chunk_length] \
                #         for i in range(0, len(labels), chunk_length)]
                

                return targets, labels

        return WrapperDataset

    def prepare_seq(self, seq: List[int]) -> List[int]:
        """
        Expand start indices into consecutive frame ranges of length sequence_length.
        """
        return [j for i in seq for j in range(i, i + self.chunk_length)]

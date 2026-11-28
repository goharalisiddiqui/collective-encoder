from typing import List, Tuple, Any, Dict, Union
import numpy as np
import ase
import torch
from collective_encoder.datareaders.processors.base import BaseDataProcessor

class FilterProcessor(BaseDataProcessor):
    """
    Filters frames based on label values.
    """
    
    _IDENTIFIER = "FILTER"
    _REQUIRED_ARGS = [
        'label_index',
        'min_val',
        'max_val'
    ]
    
    def _filter_targets(self, 
        targets: Union[List[Any], np.ndarray, torch.Tensor], 
        mask: Union[np.ndarray, List[bool]]
        ) -> Union[List[Any], np.ndarray, torch.Tensor]:

        if isinstance(targets, (np.ndarray, torch.Tensor)):
            return targets[mask]
        elif isinstance(targets, list):
            return [target for i, target in enumerate(targets) if mask[i]]
        else:
            self.raise_error(f"Unsupported target type for filtering: {type(targets)}")
    
    def postprocess_seq(self, 
        targets: Union[List[Any], np.ndarray, torch.Tensor], 
        labels: Union[List[Any], np.ndarray, torch.Tensor]
        ) -> Tuple[Union[List[Any], np.ndarray, torch.Tensor], 
                   Union[List[Any], np.ndarray, torch.Tensor]]:
    

        if isinstance(labels, (np.ndarray, torch.Tensor)) and labels.ndim == 1:
            mask = labels[:, self.label_index] > self.min_val and labels[:, self.label_index] < self.max_val
            labels = labels[mask]
            target = self._filter_targets(targets, mask)
        elif isinstance(labels, list) and isinstance(labels[0], float):
            mask = [self.min_val < label[self.label_index] < self.max_val for label in labels]
            labels = [label for i, label in enumerate(labels) if mask[i]]
            target = self._filter_targets(targets, mask)
        else:
            self.raise_error(f"Unsupported label type for filtering: {type(labels)}")

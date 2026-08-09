"""
This module provides utility functions for resolving labels and selecting specific labels from a dictionary of labels.
"""
from typing import Dict, Tuple

import numpy as np

import logging

_logger = logging.getLogger(__name__)

def cos_sin_to_angle(labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Resolves pairs of cosine and sine labels into angle labels. 
    For each label name ending with '_cos', looks for a corresponding label 
    name ending with '_sin' and combines them into a single label with the 
    original name without the suffix, containing the angle computed from the 
    cosine and sine values.
    """
    resolved_labels = {}
    for label_name, label_tensor in labels.items():
        if label_name.endswith('_cos'):
            sin_name = label_name.replace('_cos', '_sin')
            if sin_name in labels:
                cos_values = label_tensor
                sin_values = labels[sin_name]
                angles = np.arctan2(sin_values, cos_values)
                base_name = label_name[:-4]  # Remove '_cos' suffix
                resolved_labels[base_name] = angles
            else:
                _logger.warning(f"Cosine label '{label_name}' has no corresponding "
                            f"sine label '{sin_name}'. Skipping angle resolution for this label.")
        elif label_name.endswith('_sin'):
            cos_name = label_name.replace('_sin', '_cos')
            if cos_name not in labels:
                _logger.warning(f"Sine label '{label_name}' has no corresponding "
                            f"cosine label '{cos_name}'. Skipping angle resolution for this label.")
        else:
            resolved_labels[label_name] = label_tensor
    return resolved_labels

def label_selector(labels: Dict[str, np.ndarray], 
                   labels_selection_map: Dict[str, Tuple[str, int]] = None) -> Dict[str, np.ndarray]:
    """
    Selects specific labels from the provided labels dictionary based on the labels_selection_map.
    If labels_selection_map is None, returns the original labels dictionary.
    
    labels_selection_map is a dictionary where keys are the desired label names and values are tuples containing:
    - The original label name in the labels dictionary.
    - The index of the specific label to select from the original label's array.
    """
    if labels_selection_map is None:
        return labels
    selected_labels = {}
    for label_name, sel in labels_selection_map.items():
        label_ident, label_idx = sel[0], sel[1]
        if label_ident not in labels:
            raise ValueError(f"Model label '{label_ident}' specified in "
                             f"labels_selection_map not found in labels from model."
                             f" Available labels: {list(labels.keys())}.")
        if label_idx >= labels[label_ident].shape[1]:
            raise ValueError(f"Index {label_idx} for label '{label_name}' exceeds the available data length "
                             f"for '{label_ident}' (length {labels[label_ident].shape[1]}). Check the labels_selection_map and the dataset configuration.")
        selected_labels[label_name] = labels[label_ident][:, label_idx]
    return selected_labels

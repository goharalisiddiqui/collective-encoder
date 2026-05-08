import os
from typing import Dict, List
import torch

import numpy as np

from scipy.special import comb

import matplotlib.pyplot as plt

from collective_encoder.testplotters.base import BaseTestPlotter

def combinations(n, r):
    # Generate all combinations of n items taken r at a time
    pool = np.arange(n)
    indices = np.arange(r)
    yield tuple(int(pool[i]) for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(int(pool[i]) for i in indices)

class ALA2plotter(BaseTestPlotter):
    _IDENTIFIER = "ALA2plotter"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'labels_selection_map': None,  # Optional dict mapping the entries in label dict from model to that from labeler (e.g. {"psi_cos": (dihedral_cos, 6)})
    })
    
    def cossin_resolver(self, labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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
                    self.warn(f"Cosine label '{label_name}' has no corresponding "
                              f"sine label '{sin_name}'. Skipping angle resolution for this label.")
            elif label_name.endswith('_sin'):
                cos_name = label_name.replace('_sin', '_cos')
                if cos_name not in labels:
                    self.warn(f"Sine label '{label_name}' has no corresponding "
                              f"cosine label '{cos_name}'. Skipping angle resolution for this label.")
            else:
                resolved_labels[label_name] = label_tensor
        return resolved_labels

    def label_selector(self, labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.labels_selection_map is None:
            return labels
        selected_labels = {}
        for label_name, sel in self.labels_selection_map.items():
            label_ident, label_idx = sel[0], sel[1]
            if label_ident not in labels:
                self.raise_error(f"Model label '{label_ident}' specified in "
                                 f"labels_selection_map not found in labels from model.")
            selected_labels[label_name] = labels[label_ident][:, label_idx]
        return selected_labels
    
    def collection_list(self) -> List[str]:
        return ["latent", "labels", "pred"]
        
    def plot(self, data, latent, pred, labels, meta) -> None:
        labels = self.label_selector(labels)
        labels = self.cossin_resolver(labels)
        
        pred = self.label_selector(pred)
        pred = self.cossin_resolver(pred)
        
        fig, _ = self.plot_2ddihedral(pred['phi_2'], pred['psi_2'])
        self.log_image(fig, "dihedral_predictions")
        plt.close(fig)
        
        fig, _ = self.plot_2ddihedral(labels['phi_2'], labels['psi_2'])
        self.log_image(fig, "dihedral_labels")
        plt.close(fig)
        
        self.log_info(f"Plots saved in {self.outpath}")
        
from typing import Dict, List

import numpy as np

import matplotlib.pyplot as plt

from collective_encoder.testplotters.base import BaseTestPlotter
from .labels_selector import label_selector, cos_sin_to_angle


class ALA2plotter(BaseTestPlotter):
    _IDENTIFIER = "ALA2plotter"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'labels_selection_map': None,  # Optional dict mapping the entries in label dict from model to that from labeler (e.g. {"psi_cos": (dihedral_cos, 6)})
    })
    
    def collection_list(self) -> List[str]:
        return ["latent", "labels", "pred"]
        
    def plot(self, data, latent, pred, labels, meta) -> None:
        labels = label_selector(labels, self.labels_selection_map)
        labels = cos_sin_to_angle(labels)
        
        pred = label_selector(pred, self.labels_selection_map)
        pred = cos_sin_to_angle(pred)
        
        fig, _ = self.plot_2ddihedral(pred['phi_2'], pred['psi_2'])
        self.log_image(fig, "dihedral_predictions")
        plt.close(fig)
        
        fig, _ = self.plot_2ddihedral(labels['phi_2'], labels['psi_2'])
        self.log_image(fig, "dihedral_labels")
        plt.close(fig)
        
        self.log_info(f"Plots saved in {self.outpath}")
        
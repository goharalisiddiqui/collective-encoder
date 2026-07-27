import os
from typing import Dict, List
import torch

import numpy as np

import matplotlib.pyplot as plt

from .base import BaseTestPlotter
from .labels_selector import cos_sin_to_angle, label_selector

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

class LDplotter(BaseTestPlotter):
    _IDENTIFIER = "LDplotter"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'labels_selection_map': None,  # Optional dict mapping the entries in label dict from model to that from labeler (e.g. {"psi_cos": (dihedral_cos, 6)})
    })
    
    def collection_list(self) -> List[str]:
        return ["latent", "labels", "meta"]
        
    def plot(self, data, latent, pred, labels, meta) -> None:
        labels = label_selector(labels, self.labels_selection_map)
        labels = cos_sin_to_angle(labels)
        
        latent = latent.detach().cpu().numpy() if isinstance(latent, torch.Tensor) else latent
        
        self._plot_latent(latent, labels = labels, name = "latent")
        
        mu_latent = meta.get('mu_latent', None)
        if mu_latent is not None:
            mu_latent = mu_latent.detach().cpu().numpy() if isinstance(mu_latent, torch.Tensor) else mu_latent
            self._plot_latent(mu_latent, labels = labels, name = "mu_latent")

        logvar_latent = meta.get('logvar_latent', None)
        if logvar_latent is not None:
            logvar_latent = logvar_latent.detach().cpu().numpy() if isinstance(logvar_latent, torch.Tensor) else logvar_latent
            std_latent = np.sqrt(np.exp(logvar_latent))
            self._plot_latent(mu_latent, errors = std_latent, 
                             labels = labels, name = "std_latent")

        ld_names = [f"LD{i}" for i in range(latent.shape[1])]
        ref_latent = mu_latent if mu_latent is not None else latent

        fig, axes = self.plot_correlation(ref_latent, ref_latent,
                                    x_labels=ld_names, y_labels=ld_names)
        self.log_image(fig, "latent_latent_correlation")
        plt.close(fig)

        if labels is not None and len(labels) > 0:
            label_array = np.stack(list(labels.values()), axis=1)
            label_names = list(labels.keys())
            fig, axes = self.plot_correlation(ref_latent, label_array,
                                        x_labels=ld_names, y_labels=label_names)
            self.log_image(fig, "latent_label_correlation")
            plt.close(fig)

        self.log_info(f"Plots saved in {self.outpath}")
            
    def _plot_latent(self, 
                    latent, 
                    labels, 
                    errors = None, 
                    name = "latent"):
        nld = latent.shape[1]
        if nld == 1:
            fig, _ = self.plot_2dline(latent[:, 0], labels=labels, tag="LDplotter")
            self.log_image(fig, name)
        elif nld == 2:
            if errors is not None:
                fig, _ = self.plot_2dscatter(latent[:, 0], latent[:, 1], 
                                          xerr=errors[:, 0], yerr=errors[:, 1], 
                                          labels=labels, tag="0_1")
            else:
                fig, _ = self.plot_2dscatter(latent[:, 0], latent[:, 1], 
                                          labels=labels, tag="0_1")
            self.log_image(fig, f"{name}_0_1")
            plt.close(fig)
        else:
            combs = combinations(nld, 2)
            for (i, j) in combs:
                if errors is not None:
                    fig, _ = self.plot_2dscatter(latent[:, i], latent[:, j], 
                                              xerr=errors[:, i], yerr=errors[:, j], 
                                              labels=labels, tag=f"{i}_{j}")
                else:
                    fig, _ = self.plot_2dscatter(latent[:, i], latent[:, j], 
                                              labels=labels, tag=f"{i}_{j}")
                self.log_image(fig, f'{name}_{i}_{j}')
                plt.close(fig)
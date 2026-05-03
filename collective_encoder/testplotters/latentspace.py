import os
import torch

import numpy as np

from scipy.special import comb

import matplotlib.pyplot as plt

from .base import BaseTestPlotter

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
    
    def plot(self, data, latent, pred, labels, meta) -> None:
        if labels is not None:
            for k in labels.keys():
                labels[k] = labels[k].cpu().numpy()

        mu_latent = meta.get('mu_latent', None)
        if mu_latent is not None:
            mu_latent = mu_latent.detach().cpu().numpy()
            self.plot_latent(mu_latent, labels = labels, name = "mu_latent")
        logvar_latent = meta.get('logvar_latent', None)
        if logvar_latent is not None:
            logvar_latent = logvar_latent.detach().cpu().numpy()
            std_latent = np.sqrt(np.exp(logvar_latent))
            self.plot_latent(mu_latent, errors = std_latent, labels = labels, name = "std_latent")
            
        return  
    
    def plot_latent(self, 
                    latent, 
                    labels, 
                    errors = None, 
                    name = "mu_latent"):
        nld = latent.shape[1]
        if nld == 1:
            fig = self.plot_2dline(latent[:, 0], labels=labels, tag="LDplotter")
            self.log_image(fig, name)
        elif nld == 2:
            if errors is not None:
                fig = self.plot_2dscatter(latent[:, 0], latent[:, 1], 
                                          xerr=errors[:, 0], yerr=errors[:, 1], 
                                          labels=labels, tag="0_1")
            else:
                fig = self.plot_2dscatter(latent[:, 0], latent[:, 1], 
                                          labels=labels, tag="0_1")
            self.log_image(fig, f"{name}_0_1")
            plt.close(fig)
        else:
            combs = combinations(nld, 2)
            for (i, j) in combs:
                if errors is not None:
                    fig = self.plot_2dscatter(latent[:, i], latent[:, j], 
                                              xerr=errors[:, i], yerr=errors[:, j], 
                                              labels=labels, tag=f"{i}_{j}")
                else:
                    fig = self.plot_2dscatter(latent[:, i], latent[:, j], 
                                              labels=labels, tag=f"{i}_{j}")
                self.log_image(fig, f"{name}_{i}_{j}")
                plt.close(fig)
import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from scipy.special import comb

import matplotlib.pyplot as plt

from collective_encoder.common.module import CEModule

try:
    import wandb
except ImportError:
    _WANDB_AVAILABLE = False
else:
    _WANDB_AVAILABLE = True

class BaseTestPlotter(CEModule, ABC):
    _IDENTIFIER = ""
    _OPTIONAL_ARGS = {
        'logger': None,
    }
    
    def __init__(self, 
                 args, 
                 **kwargs):
        super().__init__(args, **kwargs)
        self.outpath = os.path.join(self.run_dir, type(self).__name__+"_plots")
        os.makedirs(self.outpath, exist_ok=True)
        
        logger_type = None
        if self.logger is not None:
            logger_type = type(self.logger).__name__
            if logger_type == "WandbLogger":
                if not _WANDB_AVAILABLE:
                    self.log_warn("WandbLogger is specified but wandb package is not available. "
                                  "Please install wandb to enable logging to WandbLogger.")
            else:
                self.log_warn(f"Logger is unknown type {logger_type}, "
                              f"cannot log image to logger.")
        self.logger_type = logger_type
        self.log_info(f"Initialized {type(self).__name__} with logger of "
                      f"type {logger_type} and output path {self.outpath}")

    @abstractmethod
    def plot(self, data, latent, pred, labels, meta) -> None:
        pass  
    
    def log_image(self, fig, name):
        fn = os.path.join(self.outpath, f"{name}.png")
        fig.savefig(fn, dpi=150)
        if self.logger_type == "WandbLogger":
            self.logger.experiment.log({
                f"[{type(self).__name__}] {name}": wandb.Image(fn)})
    
    def plot_2dscatter(self, x: np.ndarray, y: np.ndarray,
                       xerr: np.ndarray=None, yerr: np.ndarray=None,
                       labels: Dict[str, np.ndarray]=None, tag: str=None):
        if x.shape != y.shape:
            self.raise_error("x and y must have the same shape")
        if len(x.shape) != 1:
            self.raise_error("x and y must be 1D arrays")
        if labels is None or len(labels) == 0:
            self.raise_error("Labels must be provided for 2D scatter plot")
        if xerr is not None:
            if yerr is None:
                 self.raise_error("If xerr is provided, yerr must also be provided")
            if xerr.shape != x.shape or yerr.shape != y.shape:
                self.raise_error("xerr and yerr must have the same shape as x and y")
        ncols = len(labels)
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8 * ncols, 6))
        if ncols == 1:
            axes = [axes]
        for ind, (name, value) in enumerate(labels.items()):
            if len(value.shape) != 1 or value.shape[0] != x.shape[0]:
                self.raise_error(f"Label {name} must be a 1D array with the same length as x and y")
            scatter = axes[ind].scatter(x, y, c=value, cmap='viridis', alpha=0.7)
            if xerr is not None:
                axes[ind].errorbar(x, y, 
                                   xerr=xerr, yerr=yerr, 
                                   fmt='o', c='gray', 
                                   alpha=0.5, ecolor='lightgray', 
                                   elinewidth=1, capsize=2)
            axes[ind].set_xlabel(f"$LD_{tag.split('_')[0]}$")
            axes[ind].set_ylabel(f"$LD_{tag.split('_')[1]}$")
            fig.colorbar(scatter, ax=axes[ind], label=name)
        plt.tight_layout()
        return fig

    def plot_2dline(self, x, labels = None):
        if len(x.shape) != 1:
            self.raise_error("x must be a 1D array")
        if labels is None or len(labels) == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x, marker='o', linestyle='-', markersize=4)
            ax.set_xlabel("index")
            ax.set_ylabel("LD")
        else:
            ncols = len(labels)
            fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8, 6))
            if ncols == 1:
                axes = [axes]
            for ind, ax in enumerate(axes):
                label_name = list(labels.keys())[ind]
                label = labels[label_name]
                if len(label.shape) != 1 or label.shape[0] != x.shape[0]:
                    self.raise_error(f"Label {label_name} must be a 1D array with the same length as x")
                ax.scatter(x, label)
                ax.set_xlabel(f"$LD$")
                ax.set_ylabel(label)
        plt.tight_layout()
        return fig
        

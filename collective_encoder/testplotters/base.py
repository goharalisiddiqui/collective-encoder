import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from scipy.special import comb

import matplotlib.pyplot as plt
import torch

from collective_encoder.common.module import CEModule

try:
    import wandb
except ImportError:
    _WANDB_AVAILABLE = False
else:
    _WANDB_AVAILABLE = True

class BaseTestPlotter(CEModule, ABC):
    """
    Base class for test plotters. 
    This class provides a framework for collecting data in batches during testing and generating plots at the end. 
    It supports logging to various loggers, including WandbLogger if available.
    
    During each testing batch, the `add_batch` method is called to collect data, latent representations, predictions, labels, and metadata.
    The `finish` method is called at the end of testing to generate plots based on the collected data. 
    Subclasses must implement the `collection_list` and `plot` methods to specify which data to collect and how to plot it, respectively.
    
    Plots are saved to a directory named after the plotter class in the run directory.
    
    Attributes:
        logger_type (str): Type of logger to use. Currently supports WandbLogger if available.
    """

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
    
    def create_data_path(self):
        if not hasattr(self, "datapath"):
            datapath = os.path.join(self.run_dir, type(self).__name__+"_data")
            os.makedirs(datapath, exist_ok=True)
            self.datapath = datapath
            self.log_info(f"Created data path at {self.datapath}")
    
    @abstractmethod
    def collection_list(self) -> List[str]:
        """
        Returns a list of data types to collect during testing.
        Can include any combination of "data", "latent", "pred", "labels", and "meta".
        """
        pass
    
    @abstractmethod
    def plot(self, data, latent, pred, labels, meta) -> None:
        """
        Generates plots based on the collected data.
        This method is called at the end of testing after all batches have been processed.
        Subclasses must implement this method to define how to plot the collected data.
        
        Arguments are 'data', 'latent', 'pred', 'labels', and 'meta', which correspond 
        to the collected data types specified in `collection_list`.
        
        Arguments:
            data: Collected input data.
            latent: Collected latent representations.
            pred: Collected predictions.
            labels: Collected labels.
            meta: Collected metadata.
        """
        pass
    
    def _convert_data(self, data):
        """
        Converts data to a numpy array. If the data is a torch.Tensor, it is moved to CPU and converted to numpy.
        If the data is not a torch.Tensor, it is converted to a numpy array using np.asarray. The resulting array is ensured to be at least 1D using np.atleast_1d.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        else:
            try:
                data = np.asarray(data)
            except (ValueError, TypeError):
                self.log_error(f"Failed to convert data of type {type(data)} to numpy array.")
        return np.atleast_1d(data)

    def _collect_data(self, data, name):
        if isinstance(data, dict):
            if not hasattr(self, f"collected_{name}"):
                converted = {
                    k: self._convert_data(v)
                    for k, v in data.items()
                }
                setattr(self, f"collected_{name}", converted)
            else:
                for k in data.keys():
                    data_np = self._convert_data(data[k])
                    if k in getattr(self, f"collected_{name}"):
                        getattr(self, f"collected_{name}")[k] = np.concatenate(
                            (getattr(self, f"collected_{name}")[k], data_np), axis=0)
                    else:
                        getattr(self, f"collected_{name}")[k] = data_np
        else:
            data_np = self._convert_data(data)
            if not hasattr(self, f"collected_{name}"):
                setattr(self, f"collected_{name}", data_np)
            else:
                setattr(self, f"collected_{name}", np.concatenate(
                    (getattr(self, f"collected_{name}"), data_np), axis=0))
    
    def add_batch(self, data, latent, pred, labels, meta):
        collection_list = self.collection_list()
        if "data" in collection_list:
            self._collect_data(data, "data")
        if "latent" in collection_list:
            self._collect_data(latent, "latent")
        if "pred" in collection_list:
            self._collect_data(pred, "pred")
        if "labels" in collection_list:
            self._collect_data(labels, "labels")
        if "meta" in collection_list:
            self._collect_data(meta, "meta")
    
    def finish(self) -> None:
        self.plot(
            data=getattr(self, "collected_data", None),
            latent=getattr(self, "collected_latent", None),
            pred=getattr(self, "collected_pred", None),
            labels=getattr(self, "collected_labels", None),
            meta=getattr(self, "collected_meta", None)
        )

    def save_data(self, data, name):
        if not isinstance(data, np.ndarray):
            self.raise_error("Data must be a numpy array to be saved.")
        self.create_data_path()
        fn = os.path.join(self.datapath, f"{name}.npy")
        np.save(fn, data)
        self.log_info(f"Saved data '{name}' to {fn}")

    def log_image(self, fig, name):
        fn = os.path.join(self.outpath, f"{name}.png")
        fig.savefig(fn, dpi=150)
        if self.logger_type == "WandbLogger":
            self.logger.experiment.log({
                f"[{type(self).__name__}] {name}": wandb.Image(fn)})
        self.log_info(f"Saved plot '{name}' to {fn}")
    
    def plot_2ddihedral(self, x: np.ndarray, y: np.ndarray) -> Tuple[plt.Figure, List[plt.Axes]]:
        fig, axes = self.plot_2dscatter(x, y, labels=None)
        axes[0].set_xlabel(r"$\phi$ (radians)")
        axes[0].set_ylabel(r"$\psi$ (radians)")
        axes[0].set_xlim([-np.pi, np.pi])
        axes[0].set_ylim([-np.pi, np.pi])
        axes[0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axes[0].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        axes[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axes[0].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        
        return fig, axes
    
    def plot_matrix(self, 
                    matrix: np.ndarray, 
                    label: str, 
                    tag = "Source_Target") -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel(f"{tag.split('_')[1]} State")
        ax.set_ylabel(f"{tag.split('_')[0]} State")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_xticklabels([str(i) for i in range(matrix.shape[1])], rotation=45, ha='right')
        ax.set_yticklabels([str(i) for i in range(matrix.shape[0])])
        plt.tight_layout()

        return fig, ax
    
    def plot_2dscatter(self, x: np.ndarray, y: np.ndarray,
                       xerr: np.ndarray=None, yerr: np.ndarray=None,
                       labels: Dict[str, np.ndarray]=None, tag: str="0_1") -> Tuple[plt.Figure, List[plt.Axes]]:
        if x.shape != y.shape:
            self.raise_error("x and y must have the same shape")
        if len(x.shape) != 1:
            self.raise_error("x and y must be 1D arrays")
        if xerr is not None:
            if yerr is None:
                 self.raise_error("If xerr is provided, yerr must also be provided")
            if xerr.shape != x.shape or yerr.shape != y.shape:
                self.raise_error("xerr and yerr must have the same shape as x and y")
        if labels is None or len(labels) == 0:
            labels = {"": None}
        ncols = len(labels)
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8 * ncols, 6))
        if ncols == 1:
            axes = [axes]
        for ind, (name, value) in enumerate(labels.items()):
            if not isinstance(value, np.ndarray) or (len(value.shape) != 1 or value.shape[0] != x.shape[0]):
                self.raise_error(f"Label {name} must be a 1D array with the same length as x and y")
            scatter = axes[ind].scatter(x, y, 
                                        c=value, 
                                        cmap='viridis' if value is not None else None, 
                                        alpha=0.7)
            if xerr is not None:
                axes[ind].errorbar(x, y, 
                                   xerr=xerr, yerr=yerr, 
                                   fmt='o', c='gray', 
                                   alpha=0.5, ecolor='lightgray', 
                                   elinewidth=1, capsize=2)
            axes[ind].set_xlabel(f"$LD_{tag.split('_')[0]}$")
            axes[ind].set_ylabel(f"$LD_{tag.split('_')[1]}$")
            if value is not None:
                fig.colorbar(scatter, ax=axes[ind], label=name)
        plt.tight_layout()
        return fig, axes

    def plot_correlation(self, x: np.ndarray, y: np.ndarray,
                         x_labels: list = None, y_labels: list = None) -> Tuple[plt.Figure, plt.Axes]:
        if x.ndim != 2 or y.ndim != 2:
            self.raise_error("x and y must be 2D arrays")
        if x.shape[0] != y.shape[0]:
            self.raise_error("x and y must have the same number of samples")
        n_x, n_y = x.shape[1], y.shape[1]

        combined = np.hstack([x, y])
        full_corr = np.corrcoef(combined.T)
        corr_matrix = full_corr[:n_x, n_x:]  # (n_x, n_y) cross-correlation block

        if x_labels is None:
            x_labels = [str(i) for i in range(n_x)]
        if y_labels is None:
            y_labels = [str(j) for j in range(n_y)]

        fig, ax = plt.subplots(figsize=(max(4, n_y * 1.2), max(3, n_x * 0.8)))
        im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        fig.colorbar(im, ax=ax, label='Pearson r')

        ax.set_yticks(range(n_x))
        ax.set_yticklabels(x_labels)
        ax.set_xticks(range(n_y))
        ax.set_xticklabels(y_labels, rotation=45, ha='right')

        for i in range(n_x):
            for j in range(n_y):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(corr_matrix[i, j]) > 0.7 else 'black')
        plt.tight_layout()
        return fig, ax

    def plot_2dline(self, x, labels = None):
        if len(x.shape) != 1:
            self.raise_error("x must be a 1D array")
        if labels is None or len(labels) == 0:
            fig, axes = plt.subplots(figsize=(8, 6))
            axes.plot(x, marker='o', linestyle='-', markersize=4)
            axes.set_xlabel("index")
            axes.set_ylabel("LD")
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
        return fig, axes
        

import os
from abc import ABC
from typing import Any, Dict, List, Tuple

import numpy as npp
import numpy as np
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import torch

from collective_encoder.common.module import CEModule
from collective_encoder.common.transforms import add_transformed
from collective_encoder.utils import check_dict_contains_keys, get_missing_keys

try:
    import wandb
except ImportError:
    _WANDB_AVAILABLE = False
else:
    _WANDB_AVAILABLE = True

class BasePlotter(CEModule, ABC):
    """
    Abstract base class for all plotters.

    Test plotters accumulate data (inputs, latents, outputs, labels) during
    evaluation batches and generate visualizations and metrics at the end.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary for the plotter.
    kwargs : dict
        Additional keyword arguments (e.g. for `CEModule`).
    """

    _IDENTIFIER = ""
    _OPTIONAL_ARGS = {
        'logger': None,
        'transformed_values': None,
        'plots_2dscatter_cb': [],
        'correlations': [],
    }
    _PLOTTER_TYPE = ""
    
    
    def __init__(self, 
                 args, 
                 **kwargs):
        super().__init__(args, **kwargs)
        self.output_dir = self.safe_create_dir(
          os.path.join(self.run_dir, 
                       f"{self._PLOTTER_TYPE}_{self._IDENTIFIER.lower()}")
        )
        self.outpath = self.output_dir
        self.run_dir = self.output_dir
        
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

    ############################################################################
    # Utilities
    ############################################################################
    
    def _convert_data(self, data, ensure_1d=True):
        """
        Converts data to a numpy array. If the data is a torch.Tensor, it is moved to CPU and converted to numpy.
        If the data is not a torch.Tensor, it is converted to a numpy array using np.asarray. The resulting array is ensured to be at least 1D using np.atleast_1d.
        """
        if isinstance(data, np.ndarray):
            return np.atleast_1d(data)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        else:
            try:
                data = np.asarray(data)
            except (ValueError, TypeError):
                self.log_error(f"Failed to convert data of type {type(data)} to numpy array.")
        return np.atleast_1d(data) if ensure_1d else data
    
    def _convert_to_numpy(self, data):
        return self._convert_data(data, ensure_1d=False) 
        
    def cos_sin_to_angle(self, labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Resolves pairs of cosine and sine labels into angle labels. 

        For each label name ending with '_cos', looks for a corresponding label 
        name ending with '_sin' and combines them into a single label with the 
        original name without the suffix, containing the angle computed from the 
        cosine and sine values.

        Parameters
        ----------
        labels : dict of str to numpy.ndarray
            Dictionary of label arrays.

        Returns
        -------
        dict
            Dictionary containing resolved angular labels and other unchanged labels.
        """
        resolved_labels = {}
        for label_name, label_tensor in labels.items():
            if label_name.endswith('_cos'):
                sin_name = label_name.replace('_cos', '_sin')
                if sin_name in labels:
                    cos_values = label_tensor
                    sin_values = labels[sin_name]
                    if np.any([cos_values is None, sin_values is None]):
                        self.log_warn(f"Cosine or sine label '{label_name}' or '{sin_name}' is None. Skipping angle resolution for this label.")
                        continue
                    angles = np.arctan2(sin_values, cos_values)
                    base_name = label_name[:-4]  # Remove '_cos' suffix
                    resolved_labels[base_name] = angles
                else:
                    self.log_warn(f"Cosine label '{label_name}' has no corresponding "
                                f"sine label '{sin_name}'. Skipping angle resolution for this label.")
            elif label_name.endswith('_sin'):
                cos_name = label_name.replace('_sin', '_cos')
                if cos_name not in labels:
                    self.log_warn(f"Sine label '{label_name}' has no corresponding "
                                f"cosine label '{cos_name}'. Skipping angle resolution for this label.")
            else:
                resolved_labels[label_name] = label_tensor
        return resolved_labels
        

    ############################################################################
    # Functions for Selection Parsing and Validation
    ############################################################################
    
    def _check_selections_validity(self, selection, data, dataname):
        """
        Validates that the provided selection indexing applies safely to the data.

        Parameters
        ----------
        selection : dict or None
            A dictionary defining how to slice or select elements.
        data : numpy.ndarray or dict
            The data to validate the selection against.
        dataname : str
            A string identifier (e.g. 'labels', 'latent') for error messages.
        """
        if selection is None:
            return
        
        if type(selection) is not dict:
            self.log_error(f"{selection} must be a dictionary.")

        sel_type = type(selection.values().__iter__().__next__())
        if sel_type not in [int, str, list]:
            self.log_error(f"Selection values must be either int or list.")
        for value in selection.values():
            if not isinstance(value, sel_type):
                raise ValueError(f"All values in selection must be of the same type: {sel_type}.")

        if sel_type is int:
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Data must be a numpy array when selection values are integers.")
            if len(data.shape) < 2:
                raise ValueError(f"Data must be at least 2D to select an index."
                                 f" Found shape: {data.shape} for data '{dataname}'.")
            for idx in selection.values():
                if idx >= data.shape[1]:
                    raise ValueError(f"Index {idx} exceeds the available data length "
                                        f"(length {data.shape[1]}).")
        elif sel_type in [list, str]:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary when selection values are strings or lists."
                                 f" Found type: {type(data)} for data '{dataname}'.")
            if sel_type is str:
                for key in selection.values():
                    if key not in data:
                        raise ValueError(f"Key '{key}' from selection not found in data."
                                            f"Available keys: {list(data.keys())}.")
            else:  # sel_type is list
                for value in selection.values():
                    if len(value) != 2 or not isinstance(value[0], str) or not isinstance(value[1], int):
                        raise ValueError(f"List values must be of the form (str, int).")
                    key, idx = value[0], value[1]
                    if key not in data:
                        raise ValueError(f"Key '{key}' from selection not found in '{dataname}'.")
                    if len(data[key].shape) < 2:
                        raise ValueError(f"Data for key '{key}' must be at least 2D to select an index. "
                                         f"Found shape: {data[key].shape} for key '{key}' in data.")
                    if idx >= data[key].shape[1]:
                        raise ValueError(f"Index {idx} for key '{key}' exceeds the available data length "
                                            f"(length {data[key].shape[1]}).")
    
    def _parse_selection(self, selection, data, dataname):
        """
        Apply index-based selections to filter components of the accumulated data.

        Parameters
        ----------
        selection : dict
            Selection dictionary.
        data : numpy.ndarray or dict
            The source data tensor or dict.
        dataname : str
            Name of the data component (used in error messages).

        Returns
        -------
        dict
            A dictionary mapping the selected keys to 1D numpy arrays.
        """
        if selection is None:
            return {}
        self._check_selections_validity(selection, data, dataname)
        sel_type = type(selection.values().__iter__().__next__())
        try:
            if sel_type is int:
                return {key: self._convert_to_numpy(data[:, idx]) for key, idx in selection.items()}
            elif sel_type in [list, str]:
                selected_data = {}
                for key, value in selection.items():
                    if sel_type is str:
                        selected_data[key] = self._convert_to_numpy(data[value])
                    else:  # sel_type is list
                        selected_data[key] = self._convert_to_numpy(data[value[0]][:, value[1]])
                selected_data = self.cos_sin_to_angle(selected_data)
                return selected_data
            else:
                raise ValueError(f"Unsupported selection type: {sel_type}")
        except Exception as e:
            self.log_exception(f"Error while parsing selection for {dataname}: {e}")
            raise

    ############################################################################
    # Helper Methods for Plotting and Saving Data
    ############################################################################
    
    def _create_data_path(self):
        """
        Create a subdirectory for raw numpy arrays if it does not exist.
        """
        if not hasattr(self, "datapath"):
            datapath = os.path.join(self.outpath, "data")
            os.makedirs(datapath, exist_ok=True)
            self.datapath = datapath
            self.log_info(f"Created data path at {self.datapath}")

    def _save_data(self, data, name):
        """
        Save a numpy array to the plotter's data subdirectory.

        Parameters
        ----------
        data : numpy.ndarray
            The array to save.
        name : str
            The base filename for the saved array.
        """
        if not isinstance(data, np.ndarray):
            self.raise_error("Data must be a numpy array to be saved.")
        self._create_data_path()
        fn = os.path.join(self.datapath, f"{name}.npy")
        np.save(fn, data)
        self.log_info(f"Saved data '{name}' to {fn}")


    def log_image(self, fig, name, subpath=None):
        """
        Save a matplotlib figure to disk and log it to an active experiment logger (e.g. wandb).

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        name : str
            Base filename for the image.
        subpath : str, optional
            Subdirectory inside the main output path.
        """
        if subpath is not None:
            fn = os.path.join(self.outpath, subpath, f"{name}.png")
        else:
            fn = os.path.join(self.outpath, f"{name}.png")
        fig.savefig(fn, dpi=150)
        if self.logger_type == "WandbLogger":
            self.logger.experiment.log({
                f"[{type(self).__name__}] {name}": wandb.Image(fn)})
        self.log_info(f"Saved plot '{name}' to {fn}")
    
    def _plot_2ddihedral(self, x: np.ndarray, y: np.ndarray) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Generate a 2D scatter plot configured with dihedral axes (-pi to pi).

        Parameters
        ----------
        x : numpy.ndarray
            X coordinates (radians).
        y : numpy.ndarray
            Y coordinates (radians).

        Returns
        -------
        tuple
            (Figure, List of Axes).
        """
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
    
    def _plot_matrix(self, 
                    matrix: np.ndarray, 
                    label: str, 
                    tag = "Source_Target") -> Tuple[plt.Figure, plt.Axes]:
        """
        Generate a 2D heatmap image representing a matrix (e.g. transition or correlation).

        Parameters
        ----------
        matrix : numpy.ndarray
            2D numeric matrix.
        label : str
            Colorbar label.
        tag : str, optional
            A string formatted as "Y-Axis_X-Axis" to label the axes.

        Returns
        -------
        tuple
            (Figure, Axes).
        """
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

    ############################################################################
    # 2D Scatter Plot
    ############################################################################
    
    def _plot_line(self, x, labels = None):
        """
        Generate 1D/2D line plots for evaluating latent features.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of values (e.g. frame indices or a selected variable).
        labels : dict, optional
            A dictionary mapping label names to arrays to plot against x.

        Returns
        -------
        tuple
            (Figure, List of Axes).
        """
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
    
    plot_2dline = _plot_line # Alias for backward compatibility with older code that used `plot_2dline` instead of `_plot_line`.
    
    ############################################################################
    # 2D Scatter Plot
    ############################################################################

    def _plot_2dscatter(self, x: np.ndarray, y: np.ndarray,
                        xerr: np.ndarray=None, yerr: np.ndarray=None,
                        xlabel: str="X", ylabel: str="Y",
                        clabels: np.ndarray=None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Generate 2D scatter plots of x vs y, optionally colored by various labels.

        Parameters
        ----------
        x : numpy.ndarray
            X-axis points.
        y : numpy.ndarray
            Y-axis points.
        xerr : numpy.ndarray, optional
            Error bars for X.
        yerr : numpy.ndarray, optional
            Error bars for Y.
        xlabel : str, optional
            Label for X-axis.
        ylabel : str, optional
            Label for Y-axis.
        clabels : dict, optional
            A dictionary mapping label names to their values (used for point coloring).

        Returns
        -------
        tuple
            (Figure, List of Axes).
        """
        if x.shape != y.shape:
            self.raise_error("x and y must have the same shape")
        if len(x.shape) != 1:
            self.raise_error("x and y must be 1D arrays")
        if xerr is not None:
            if yerr is None:
                self.raise_error("If xerr is provided, yerr must also be provided")
            if xerr.shape != x.shape or yerr.shape != y.shape:
                self.raise_error(f"xerr and yerr must have the same shape as x and y "
                                 f"xerr.shape: {xerr.shape}, "
                                 f"yerr.shape: {yerr.shape}, "
                                 f"x.shape: {x.shape}, "
                                 f"y.shape: {y.shape}")
        if clabels is None or len(clabels) == 0:
            clabels = {"": None}
        ncols = len(clabels)
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8 * ncols, 6))
        if ncols == 1:
            axes = [axes]
        for ind, (name, value) in enumerate(clabels.items()):
            if value is not None and ( \
                not isinstance(value, np.ndarray) \
                    or (len(value.shape) != 1 \
                        or value.shape[0] != x.shape[0])):
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
            axes[ind].set_xlabel(xlabel)
            axes[ind].set_ylabel(ylabel)
            if value is not None:
                fig.colorbar(scatter, ax=axes[ind], label=name)
                
        plt.tight_layout()
        
        return fig, axes
    
    def after_2dscatter(self, 
                        plot : Dict[str, Any],
                        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]], # x, y, xerr, yerr, labels
                        fig: plt.Figure,
                        axes: np.ndarray
                        ) -> None:
        """
        Hook for additional processing after a 2D scatter plot is generated. Can be overridden in subclasses to add custom behavior.
        """
        pass

    @staticmethod
    def _prepare_scatter_1d(
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_err: np.ndarray = None,
        y_err: np.ndarray = None,
        color_data: Dict[str, np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Flatten multi-dimensional ``(N_samples, D)`` scatter arrays to 1-D pairs,
        broadcasting per-sample 1-D color arrays across features and filtering
        out non-finite (``NaN`` / ``inf``) entries.
        """
        x_arr = np.asarray(x_data)
        y_arr = np.asarray(y_data)
        if x_arr.ndim > 1 and x_arr.shape == y_arr.shape:
            n_samples = x_arr.shape[0]
            n_feats = int(np.prod(x_arr.shape[1:]))
            if color_data is not None:
                new_color = {}
                for c, arr in color_data.items():
                    c_arr = np.asarray(arr)
                    if c_arr.ndim == 2 and c_arr.shape[1] == 1:
                        c_arr = c_arr.squeeze(1)
                    if c_arr.ndim == 1 and c_arr.shape[0] == n_samples:
                        new_color[c] = np.repeat(c_arr, n_feats)
                    else:
                        new_color[c] = c_arr.reshape(-1)
                color_data = new_color
            x_arr = x_arr.reshape(-1)
            y_arr = y_arr.reshape(-1)
            if x_err is not None:
                x_err = np.asarray(x_err).reshape(-1)
            if y_err is not None:
                y_err = np.asarray(y_err).reshape(-1)
        elif color_data is not None:
            color_data = {
                c: (
                    np.asarray(arr).squeeze(1)
                    if np.asarray(arr).ndim == 2 and np.asarray(arr).shape[1] == 1
                    else np.asarray(arr).reshape(-1)
                )
                for c, arr in color_data.items()
            }

        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if not np.all(valid):
            x_arr = x_arr[valid]
            y_arr = y_arr[valid]
            if x_err is not None:
                x_err = np.asarray(x_err)[valid]
            if y_err is not None:
                y_err = np.asarray(y_err)[valid]
            if color_data is not None:
                color_data = {c: arr[valid] for c, arr in color_data.items()}

        return x_arr, y_arr, x_err, y_err, color_data

    @staticmethod
    def _decorate_scatter_axes(
        axes: List[plt.Axes],
        plot: Dict[str, Any],
        x_data: np.ndarray,
        y_data: np.ndarray,
        title_override: str = None,
    ) -> None:
        """
        Apply angular axis limits/ticks, title, and optional ``y = x`` diagonal
        ideal reference line to 2-D scatter plot axes.
        """
        is_angles = bool(plot.get("angles", False))
        is_angle_x = is_angles or bool(plot.get("angle_x", False))
        is_angle_y = is_angles or bool(plot.get("angle_y", False))
        draw_diag = bool(
            plot.get("diagonal", False)
            or plot.get("ideal_line", False)
            or plot.get("x_eq_y", False)
        )
        title = title_override if title_override is not None else plot.get("title", None)

        for ax in axes:
            if is_angle_x:
                ax.set_xlim([-np.pi, np.pi])
                ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
                ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
            if is_angle_y:
                ax.set_ylim([-np.pi, np.pi])
                ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
                ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
            if draw_diag and x_data.size > 0 and y_data.size > 0:
                if is_angle_x and is_angle_y:
                    d_min, d_max = -np.pi, np.pi
                else:
                    d_min = float(min(np.min(x_data), np.min(y_data)))
                    d_max = float(max(np.max(x_data), np.max(y_data)))
                ax.plot(
                    [d_min, d_max],
                    [d_min, d_max],
                    linestyle="--",
                    color="crimson",
                    linewidth=1.5,
                    label="Ideal (x = y)",
                )
                ax.legend(loc="upper left", fontsize="small")
            if title:
                ax.set_title(title)

    @staticmethod
    def _resolve_group_label(group_names: Any, g_val: Any) -> str:
        """Resolve a human-readable group/species name from ``group_names``."""
        try:
            g_int = int(g_val)
        except (ValueError, TypeError):
            g_int = None

        if isinstance(group_names, dict):
            if g_val in group_names:
                return str(group_names[g_val])
            if g_int is not None and g_int in group_names:
                return str(group_names[g_int])
            if str(g_val) in group_names:
                return str(group_names[str(g_val)])
        elif isinstance(group_names, (list, tuple)) and g_int is not None:
            if 0 <= g_int < len(group_names):
                return str(group_names[g_int])
        return f"species_{g_int}" if g_int is not None else str(g_val)

    def _plot_2dscatter_plots(self, vals: Dict[str, np.ndarray]) -> None:
        """
        Generate configured 2D scatter plots from the accumulated data.

        Parameters
        ----------
        vals : dict of str to numpy.ndarray
            Unified dictionary of all accumulated numeric data.
        """
        for plot in self.plots_2dscatter_cb:
            if any(key not in plot.keys() for key in ["x", "y"]):
                self.log_exception(f"Missing 'x' or 'y' in plot specification: {plot}. Skipping this plot.")
                continue
            x_label, y_label = plot['x'], plot['y']
            color_label = plot.get('color', None)
            x_error_data, y_error_data = None, None
            if 'x_error' in plot:
                x_error_label = plot['x_error']
                if x_error_label not in vals:
                    self.log_exception(f"Label '{x_error_label}' for x error not found in collected data. Available labels: {list(vals.keys())}. Skipping this plot.")
                    continue
                x_error_data = vals[x_error_label]
            if 'y_error' in plot:
                y_error_label = plot['y_error']
                if y_error_label not in vals:
                    self.log_exception(f"Label '{y_error_label}' for y error not found in collected data. Available labels: {list(vals.keys())}. Skipping this plot.")
                    continue
                y_error_data = vals[y_error_label]
            if x_label not in vals:
                self.log_exception(f"Label '{x_label}' not found in collected data. Available labels: {list(vals.keys())}.")
                continue
            if y_label not in vals:
                self.log_exception(f"Label '{y_label}' not found in collected data. Available labels: {list(vals.keys())}.")
                continue
            if color_label is not None:
                missing_colors = [c for c in color_label.split(':') if c not in vals]
                if missing_colors:
                    self.log_exception(f"Color label(s) '{missing_colors}' not found in collected data. Available labels: {list(vals.keys())}. Skipping this plot.")
                    continue
            x_data = vals[x_label]
            y_data = vals[y_label]
            color_data = {c: vals[c] for c in color_label.split(':') if c in vals} if color_label is not None else None
            group_key = plot.get("group_by", None)
            group_arr = None
            if group_key is not None:
                if group_key not in vals:
                    self.log_exception(f"Group label '{group_key}' not found in collected data. Available labels: {list(vals.keys())}. Skipping this plot.")
                    continue
                group_arr = np.asarray(vals[group_key]).reshape(-1)

            fname = plot.get('xlabel', x_label) + "_" + plot.get('ylabel', y_label)
            if 'sparse_steps' in plot:
                sparse_steps = plot['sparse_steps']
                if not isinstance(sparse_steps, int) or sparse_steps <= 0:
                    self.log_exception(f"'sparse_steps' must be a positive integer. Found: {sparse_steps}. Skipping this plot.")
                    continue
                x_data = x_data[::sparse_steps]
                y_data = y_data[::sparse_steps]
                x_error_data = x_error_data[::sparse_steps] if x_error_data is not None else None
                y_error_data = y_error_data[::sparse_steps] if y_error_data is not None else None
                color_data = {c: color_data[c][::sparse_steps] for c in color_data} if color_data is not None else None
                if group_arr is not None:
                    group_arr = group_arr[::sparse_steps]
                fname += f"_sparse{str(sparse_steps)}"

            xlabel = plot.get('xlabel', x_label)
            ylabel = plot.get('ylabel', y_label)
            base_fname = plot.get('name', fname)

            if group_arr is not None:
                uniq_groups = [g for g in np.unique(group_arr) if np.isfinite(g)]
                group_names = plot.get("group_names", None)
                group_mode = plot.get("group_mode", "subplots")
                base_title = plot.get("title", base_fname)

                if group_mode == "files":
                    for g_val in uniq_groups:
                        g_mask = group_arr == g_val
                        g_name = self._resolve_group_label(group_names, g_val)
                        gx, gy, gx_err, gy_err, gc = self._prepare_scatter_1d(
                            x_data[g_mask],
                            y_data[g_mask],
                            x_error_data[g_mask] if x_error_data is not None else None,
                            y_error_data[g_mask] if y_error_data is not None else None,
                            {c: arr[g_mask] for c, arr in color_data.items()} if color_data is not None else None,
                        )
                        if gx.size == 0:
                            continue
                        fig, axes = self._plot_2dscatter(
                            gx, gy, xerr=gx_err, yerr=gy_err,
                            xlabel=xlabel, ylabel=ylabel, clabels=gc,
                        )
                        self.after_2dscatter(plot, (gx, gy, gx_err, gy_err, gc), fig, axes)
                        self._decorate_scatter_axes(
                            axes, plot, gx, gy, title_override=f"{base_title} ({g_name})"
                        )
                        out_name = f"{base_fname}_{g_name}"
                        if tag := plot.get('tag', False):
                            out_name = f"{out_name}_{tag}"
                        self.log_image(fig, out_name)
                        plt.close(fig)
                else:
                    n_groups = max(1, len(uniq_groups))
                    fig, axes = plt.subplots(
                        nrows=1, ncols=n_groups, figsize=(6.5 * n_groups, 5.5)
                    )
                    if n_groups == 1:
                        axes = [axes]
                    for ax, g_val in zip(axes, uniq_groups):
                        g_mask = group_arr == g_val
                        g_name = self._resolve_group_label(group_names, g_val)
                        gx, gy, gx_err, gy_err, _ = self._prepare_scatter_1d(
                            x_data[g_mask],
                            y_data[g_mask],
                            x_error_data[g_mask] if x_error_data is not None else None,
                            y_error_data[g_mask] if y_error_data is not None else None,
                            None,
                        )
                        ax.scatter(gx, gy, alpha=0.7)
                        if gx_err is not None and gy_err is not None:
                            ax.errorbar(
                                gx, gy, xerr=gx_err, yerr=gy_err,
                                fmt='o', c='gray', alpha=0.5,
                                ecolor='lightgray', elinewidth=1, capsize=2,
                            )
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        self._decorate_scatter_axes(
                            [ax], plot, gx, gy, title_override=f"{base_title} ({g_name})"
                        )
                    plt.tight_layout()
                    self.after_2dscatter(plot, (x_data, y_data, x_error_data, y_error_data, color_data), fig, axes)
                    out_name = base_fname
                    if tag := plot.get('tag', False):
                        out_name = f"{out_name}_{tag}"
                    self.log_image(fig, out_name)
                    plt.close(fig)
                continue

            x_1d, y_1d, x_err_1d, y_err_1d, color_1d = self._prepare_scatter_1d(
                x_data, y_data, x_error_data, y_error_data, color_data
            )
            fig, axes = self._plot_2dscatter(x_1d, y_1d, 
                                         xerr=x_err_1d, yerr=y_err_1d,
                                         xlabel=xlabel,
                                         ylabel=ylabel,
                                         clabels=color_1d)
            self.after_2dscatter(plot, 
                (x_1d, y_1d, x_err_1d, y_err_1d, color_1d), 
                fig, axes)
            self._decorate_scatter_axes(axes, plot, x_1d, y_1d)
            out_name = base_fname
            if tag := plot.get('tag', False):
                out_name = f"{out_name}_{tag}"
            self.log_image(fig, out_name)
            plt.close(fig)
    
    ############################################################################
    # Correlations Plotting
    ############################################################################
            
    def _plot_correlation(self, x: np.ndarray, y: np.ndarray,
                         x_labels: list = None, y_labels: list = None,
                         correlation_type: str = 'spearman') -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
        """
        Plot and compute the cross-correlation matrix between datasets X and Y.

        Parameters
        ----------
        x : numpy.ndarray
            First dataset, shape (n_samples, n_features_x).
        y : numpy.ndarray
            Second dataset, shape (n_samples, n_features_y).
        x_labels : list of str, optional
            Names for the features in X.
        y_labels : list of str, optional
            Names for the features in Y.
        correlation_type : str, optional
            'spearman' or 'pearson'.

        Returns
        -------
        tuple
            (Figure, Axes, Computed Correlation Matrix).
        """
        if x.ndim != 2 or y.ndim != 2:
            self.raise_error("x and y must be 2D arrays")
        if x.shape[0] != y.shape[0]:
            self.raise_error("x and y must have the same number of samples")
        n_x, n_y = x.shape[1], y.shape[1]

        combined = np.hstack([x, y])
        if correlation_type == 'spearman':
            res = spearmanr(combined, axis=0)  # (n_x + n_y, n_x + n_y) or scalar
            stat = res.statistic if hasattr(res, 'statistic') else res[0]
            if np.ndim(stat) == 0:
                r = float(stat)
                full_corr = np.array([[1.0, r], [r, 1.0]])
            else:
                full_corr = np.atleast_2d(stat)
        elif correlation_type == 'pearson':
            full_corr = np.corrcoef(combined, rowvar=False)  # (n_x + n_y, n_x + n_y)
            full_corr = np.atleast_2d(full_corr)
        else:
            self.raise_error(f"Unsupported correlation type: {correlation_type}. Use 'spearman' or 'pearson'.")
        corr_matrix = full_corr[:n_x, n_x:]  # (n_x, n_y) cross-correlation block

        if x_labels is None:
            x_labels = [str(i) for i in range(n_x)]
        if y_labels is None:
            y_labels = [str(j) for j in range(n_y)]

        fig, ax = plt.subplots(figsize=(max(4, n_y * 1.2), max(3, n_x * 0.8)))
        im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        fig.colorbar(im, ax=ax, label=f'{correlation_type.capitalize()} Correlation')

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
        return fig, ax, corr_matrix

    def after_correlation_matrix(self, 
                                 corr : Dict[str, Any],
                                 corr_matrix: np.ndarray,
                                 fig: plt.Figure,
                                 axes: np.ndarray
                                 ) -> None:
        """
        Hook for additional processing after a correlation matrix is generated.
        """
        pass
    
    def _plot_correlations(self, vals: Dict[str, np.ndarray]) -> None:
        """
        Generate configured correlation matrices from the accumulated data.

        Parameters
        ----------
        vals : dict of str to numpy.ndarray
            Unified dictionary of all accumulated numeric data.
        """
        for corr in self.correlations:
            if not check_dict_contains_keys(corr, required_keys=['x', 'y']):
                continue
            x_labels = corr['x'].split(':')
            y_labels = corr['y'].split(':')
            if k := get_missing_keys(vals, x_labels + y_labels):
                self.log_exception(f"Missing labels in collected data: {k}. Skipping this correlation plot.")
                continue
            x_data = np.stack([vals[x] for x in x_labels], axis=1)
            y_data = np.stack([vals[y] for y in y_labels], axis=1)
            fig, axes, corr_matrix = self.plot_correlation(
                x_data, y_data,
                x_labels=x_labels, 
                y_labels=y_labels, 
                correlation_type=corr.get('type', 'spearman')
            )
            self.after_correlation_matrix(corr, corr_matrix, fig, axes)
            corr_name = corr.get('name', f"correlation_{corr['x']}_{corr['y']}")
            self.log_image(fig, corr_name)
            plt.close(fig)
            
    ############################################################################
    # Plotting
    ############################################################################
        
    def plot_vals(self) -> None:
        """
        Execute the plotting logic after all data have been collected.
        """
        if not hasattr(self, 'vals'):
            self.log_exception("No data collected for plotting. Ensure that 'vals' is populated before calling plot().")
            return
        vals = self.vals
        vals = add_transformed(self.transformed_values, vals)
        
        try:
            self._plot_2dscatter_plots(vals)
        except Exception as e:
            self.log_exception(f"Error occurred while plotting 2D scatter: {e}")

        try:
            self._plot_correlations(vals)
        except Exception as e:
            self.log_exception(f"Error occurred while plotting correlations: {e}")
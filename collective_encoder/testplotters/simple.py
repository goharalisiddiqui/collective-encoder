from typing import Any, Dict, List

import numpy as np

import matplotlib.pyplot as plt

from collective_encoder.testplotters.base import BaseTestPlotter

class SimplePlotter(BaseTestPlotter):
    """
    Standard test plotter for generating 2D scatter plots and correlation matrices.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Optional keys:
        - ``plots_2dscatter_cb`` (list of dict): Specifications for scatter plots.
          Each dict must have 'x' and 'y' keys mapping to data labels.
        - ``correlations`` (list of dict): Specifications for correlation plots.
          Each dict must have 'x' and 'y' defining the variables to correlate.
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "SimplePlotter"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    
    def collection_list(self) -> List[str]:
        """
        Define which data types need to be collected across batches.

        Returns
        -------
        list of str
            List containing ["latent", "pred", "labels", "meta"].
        """
        return ["latent", "pred", "labels", "meta"]

    def plot(self, data, latent, pred, labels, meta) -> None:
        """
        Execute the plotting logic after all batches have been collected.

        Parameters
        ----------
        data : Any
            Accumulated inputs (unused).
        latent : dict or numpy.ndarray
            Accumulated latent representations.
        pred : Any
            Accumulated predictions (unused).
        labels : dict
            Accumulated ground truth labels.
        meta : dict
            Accumulated metadata.
        """
        try:
            labels = self._parse_selection(self.labels_selection, labels, "labels")
            latent = self._parse_selection(self.latents_selection, latent, "latent")
            pred = self._parse_selection(self.pred_selection, pred, "pred")
            meta = self._parse_selection(self.meta_selection, meta, "meta")
        except Exception as e:
            self.log_exception(f"Error occurred while parsing selections: {e}")
            return

        # All names must be unique
        all_names = list(labels.keys()) + \
                    list(latent.keys()) + \
                    list(pred.keys()) + \
                    list(meta.keys())
        if len(all_names) == 0:
            self.log_exception("No labels, latent keys, or meta keys found after parsing selections. Nothing to plot.")
            return

        if len(all_names) != len(set(all_names)):
            raise ValueError(f"Duplicate names found in labels and latent keys. "
                             f"All names must be unique. Found names: {all_names}")
        self.vals = {**labels, **latent, **pred, **meta}
        
        self.plot_vals()

    def after_correlation_matrix(self, 
                                 corr : Dict[str, Any],
                                 corr_matrix: np.ndarray,
                                 fig: plt.Figure,
                                 axes: np.ndarray
                                 ) -> None:
        # Register correlation values as metrics for HPO sweeps & evaluation
        corr_name = corr.get('name', f"correlation_{corr['x']}_{corr['y']}")
        x_labels = corr['x'].split(':')
        y_labels = corr['y'].split(':')
        
        n_x, n_y = len(x_labels), len(y_labels)
        self.metrics_dict[f"{corr_name}_matrix"] = {
            "x_labels": list(x_labels),
            "y_labels": list(y_labels),
            "matrix": corr_matrix.tolist(),
        }

        if n_x == 1 and n_y == 1:
            val = float(corr_matrix[0, 0])
            self.set_metric(corr_name, val)
            self.set_metric(f"{corr_name}_abs", abs(val))

        # Determine non-self entries (ignore diagonal where x_var == y_var)
        mask = np.ones_like(corr_matrix, dtype=bool)
        for i, x_var in enumerate(x_labels):
            for j, y_var in enumerate(y_labels):
                if x_var == y_var:
                    mask[i, j] = False

        if np.any(mask):
            eval_entries = corr_matrix[mask]
        else:
            eval_entries = corr_matrix.flatten()

        eval_entries_abs = np.abs(eval_entries)
        mean_val = float(np.mean(eval_entries))
        max_val = float(np.max(eval_entries))
        min_val = float(np.min(eval_entries))
        mean_abs_val = float(np.mean(eval_entries_abs))
        max_abs_val = float(np.max(eval_entries_abs))
        min_abs_val = float(np.min(eval_entries_abs))

        self.set_metric(f"{corr_name}_mean", mean_val)
        self.set_metric(f"{corr_name}_avg", mean_val)
        self.set_metric(f"{corr_name}_max", max_val)
        self.set_metric(f"{corr_name}_min", min_val)
        self.set_metric(f"{corr_name}_mean_abs", mean_abs_val)
        self.set_metric(f"{corr_name}_avg_abs", mean_abs_val)
        self.set_metric(f"{corr_name}_max_abs", max_abs_val)
        self.set_metric(f"{corr_name}_min_abs", min_abs_val)

        for i, x_var in enumerate(x_labels):
            for j, y_var in enumerate(y_labels):
                c_val = float(corr_matrix[i, j])
                self.set_metric(f"{corr_name}_{x_var}_{y_var}", c_val)
                self.set_metric(f"{corr_name}_{x_var}_{y_var}_abs", abs(c_val))
                self.set_metric(f"{corr_name}_{i}_{j}", c_val)
                self.set_metric(f"{corr_name}_{i}_{j}_abs", abs(c_val))

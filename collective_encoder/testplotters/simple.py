from typing import Dict, List

import numpy as np

import matplotlib.pyplot as plt

from collective_encoder.testplotters.base import BaseTestPlotter
from collective_encoder.utils import check_dict_contains_keys
from collective_encoder.testplotters.utils import combinations
from collective_encoder.testplotters.transforms import add_transformed

class SimplePlotter(BaseTestPlotter):
    _IDENTIFIER = "SimplePlotter"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'plots_2dscatter_cb': [],
        'correlations': [],
    })
    
    def collection_list(self) -> List[str]:
        return ["latent", "labels", "meta"]

    def plot(self, data, latent, pred, labels, meta) -> None:
        try:
            labels = self._parse_selection(self.labels_selection, labels, "labels")
            latent = self._parse_selection(self.latents_selection, latent, "latent")
            meta = self._parse_selection(self.meta_selection, meta, "meta")
        except Exception as e:
            self.log_exception(f"Error occurred while parsing selections: {e}")
            return

        # All names must be unique
        all_names = list(labels.keys()) + \
                    list(latent.keys()) + \
                    list(meta.keys())
        if len(all_names) != len(set(all_names)):
            raise ValueError(f"Duplicate names found in labels and latent keys. "
                             f"All names must be unique. Found names: {all_names}")
        vals = {**labels, **latent, **meta}
        vals = add_transformed(self.transformed_values, vals)
        
        self._plot_2dscatter(vals)
        self._plot_correlations(vals)

    def _plot_2dscatter(self, vals: Dict[str, np.ndarray]) -> None:
        for plot in self.plots_2dscatter_cb:
            for key in ['x', 'y', 'color']:
                if key not in plot:
                    self.log_exception(f"Missing '{key}' in plot specification: {plot}. Skipping this plot.")
                    continue
            x_label, y_label, color_label = plot['x'], plot['y'], plot['color']
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
            if x_label not in vals or y_label not in vals:
                self.log_exception(f"Labels '{x_label}', '{y_label}', or '{color_label}' not found in collected data. Available labels: {list(vals.keys())}.")
                continue
            for c in color_label.split(':'):
                if c not in vals:
                    self.log_exception(f"Color label '{c}' not found in collected data. Available labels: {list(vals.keys())}. Skipping this plot.")
                    continue

            x_data = vals[x_label]
            y_data = vals[y_label]
            color_data = {c: vals[c] for c in color_label.split(':') if c in vals}
            tag = f"{x_label}_{y_label}"
            fname = tag if 'name' not in plot else plot['name']
            if 'sparse_steps' in plot:
                sparse_steps = plot['sparse_steps']
                if not isinstance(sparse_steps, int) or sparse_steps <= 0:
                    self.log_exception(f"'sparse_steps' must be a positive integer. Found: {sparse_steps}. Skipping this plot.")
                    continue
                x_data = x_data[::sparse_steps]
                y_data = y_data[::sparse_steps]
                x_error_data = x_error_data[::sparse_steps] if x_error_data is not None else None
                y_error_data = y_error_data[::sparse_steps] if y_error_data is not None else None
                color_data = {c: color_data[c][::sparse_steps] for c in color_data}
                tag += f"_sparse{str(sparse_steps)}"
            fig, _ = self.plot_2dscatter(x_data, y_data, 
                                         xerr=x_error_data, yerr=y_error_data,
                                         labels=color_data, 
                                         tag=tag)
            self.log_image(fig, fname)
            plt.close(fig)
    
    def _plot_correlations(self, vals: Dict[str, np.ndarray]) -> None:
        for corr in self.correlations:
            check_dict_contains_keys(corr, required_keys=['x', 'y'])
            x_labels = corr['x'].split(':')
            y_labels = corr['y'].split(':')
            for x in x_labels:
                if x not in vals:
                    self.log_exception(f"Correlation x label '{x}' not found in collected data. Available labels: {list(vals.keys())}. Skipping this correlation plot.")
                    continue
            for y in y_labels:
                if y not in vals:
                    self.log_exception(f"Correlation y label '{y}' not found in collected data. Available labels: {list(vals.keys())}. Skipping this correlation plot.")
                    continue
            x_data = np.stack([vals[x] for x in x_labels], axis=1)
            y_data = np.stack([vals[y] for y in y_labels], axis=1)
            fig, axes, corr_matrix = self.plot_correlation(
                x_data, y_data,
                x_labels=x_labels, 
                y_labels=y_labels, 
                correlation_type=corr.get('type', 'spearman')
            )
            corr_name = corr.get('name', f"correlation_{corr['x']}_{corr['y']}")
            self.log_image(fig, corr_name)
            plt.close(fig)

            # Register correlation values as metrics for HPO sweeps & evaluation
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

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
        labels = self._parse_selection(self.labels_selection, labels, "labels")
        latent = self._parse_selection(self.latents_selection, latent, "latent")
        meta = self._parse_selection(self.meta_selection, meta, "meta")
        
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
            fig, axes = self.plot_correlation(x_data, y_data,
                                              x_labels=x_labels, 
                                              y_labels=y_labels, 
                                              correlation_type=corr.get('type', 'spearman'))
            fname = f"correlation_{corr['x']}_{corr['y']}" if 'name' not in corr else corr['name']
            self.log_image(fig, fname)
            plt.close(fig)
        
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
from typing import Any, Dict, List, Optional

import numpy as np

from collective_encoder.testplotters.simple import SimplePlotter


class LatentCorrelationsPlotter(SimplePlotter):
    """
    Test plotter subclass of SimplePlotter that automatically selects all
    latent space dimensions as LD_1, LD_2, ..., LD_D and computes pairwise
    cross-correlations regardless of the latent dimensionality.

    Generates an imshow correlation heatmap and registers summary metrics:
      - `{name}`: 2D numpy array of all pairwise correlations (D x D)
      - `{name}_avg`: Average cross-correlation across off-diagonal elements
      - `{name}_max`: Maximum cross-correlation across off-diagonal elements
      - `{name}_min`: Minimum cross-correlation across off-diagonal elements

    Example YAML configuration:
    ```yaml
    test_plotters:
      - tester_type: LatentCorrelationsPlotter
        tester_args:
          name: "LC"
          correlation_type: "spearman"  # or "pearson"
    ```
    """

    _IDENTIFIER = "LatentCorrelationsPlotter"
    _OPTIONAL_ARGS = SimplePlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'name': 'LC',
        'correlation_type': 'spearman',
    })

    def plot(self, data, latent, pred, labels, meta) -> None:
        # Automatically select all latent dimensions as LD_1, LD_2, ... if not specified
        if self.latents_selection is None:
            if latent is not None:
                if isinstance(latent, dict):
                    self.latents_selection = {k: k for k in latent.keys()}
                else:
                    arr = np.asarray(latent)
                    n_dims = arr.shape[1] if arr.ndim >= 2 else 1
                    self.latents_selection = {f"LD_{i+1}": i for i in range(n_dims)}
            elif isinstance(meta, dict) and "mu_latent" in meta:
                arr = np.asarray(meta["mu_latent"])
                n_dims = arr.shape[1] if arr.ndim >= 2 else 1
                if self.meta_selection is None:
                    self.meta_selection = {f"LD_{i+1}": ["mu_latent", i] for i in range(n_dims)}

        # Automatically construct cross-correlation config if not specified
        name = getattr(self, "name", "LC") or "LC"
        corr_type = getattr(self, "correlation_type", "spearman")
        if not self.correlations:
            if self.latents_selection is not None:
                dim_names = list(self.latents_selection.keys())
            elif self.meta_selection is not None:
                dim_names = list(self.meta_selection.keys())
            else:
                dim_names = []

            if dim_names:
                dim_str = ":".join(dim_names)
                self.correlations = [{
                    'name': name,
                    'x': dim_str,
                    'y': dim_str,
                    'type': corr_type,
                }]

        # Run SimplePlotter plotting & correlation pipeline
        super().plot(data=data, latent=latent, pred=pred, labels=labels, meta=meta)

        # Store 2D array, save data, and log summary results
        matrix_key = f"{name}_matrix"
        if matrix_key in self.metrics_dict:
            matrix_entry = self.metrics_dict[matrix_key]
            corr_matrix = np.array(matrix_entry["matrix"])
            dim_names = matrix_entry.get("x_labels", [])
            n_dims = corr_matrix.shape[0]

            setattr(self, name, corr_matrix)
            self.LC = corr_matrix
            self.corr_matrix = corr_matrix
            self.metrics_dict[name] = corr_matrix

            self.save_data(corr_matrix, name)

            lc_avg = self.metrics_dict.get(f"{name}_avg", 0.0)
            lc_max = self.metrics_dict.get(f"{name}_max", 0.0)
            lc_min = self.metrics_dict.get(f"{name}_min", 0.0)

            self.log_result_msg("=" * 60)
            self.log_result_msg(f"Latent Correlations ({name}):")
            self.log_result_msg(f"  Dimensions: {n_dims} ({dim_names})")
            self.log_result_msg(f"  Correlation Type: {corr_type}")
            self.log_result_msg(f"  {name}_avg (Average Cross-Correlation): {lc_avg:.4f}")
            self.log_result_msg(f"  {name}_max (Maximum Cross-Correlation): {lc_max:.4f}")
            self.log_result_msg(f"  {name}_min (Minimum Cross-Correlation): {lc_min:.4f}")
            self.log_result_msg("=" * 60)

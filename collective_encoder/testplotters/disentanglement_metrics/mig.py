import os
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from collective_encoder.testplotters.disentanglement_metrics.base import BaseDisentanglementMetric


class DisentanglementMIGMetric(BaseDisentanglementMetric):
    r"""
    This implements the Mutual Information Gap (MIG) disentanglement metric.
    Theoretical basis: Chen et al., NeurIPS 2018:
    "Isolating Sources of Disentanglement in VAEs" (beta-TCVAE)
    https://openreview.net/pdf?id=BJdMRoCIf

    Key Algorithm:
    1. Discretize each continuous latent dimension z_i (i = 1..D) and each ground-truth
       generative factor v_k (k = 1..K) into discrete bins (default: num_bins = 20).
    2. Compute the empirical mutual information matrix I(z_i; v_k) and factor entropies H(v_k).
    3. For each factor v_k, sort the mutual information values across all latent dimensions:
       I(z_{j_1}; v_k) >= I(z_{j_2}; v_k) >= ... >= I(z_{j_D}; v_k).
    4. Compute the normalized mutual information gap for factor v_k:
       MIG_k = (I(z_{j_1}; v_k) - I(z_{j_2}; v_k)) / H(v_k).
    5. Overall MIG Score:
       MIG = (1 / K) \sum_{k=1}^K MIG_k \in [0, 1].
    6. Multi-split robustness: Subsample data over `num_models` random splits to report Mean ± Std & CI.
    """
    _IDENTIFIER = "DisentanglementMIGMetric"
    _OPTIONAL_ARGS = BaseDisentanglementMetric._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'num_bins': 20,                 # Number of histogram bins for discrete mutual information estimation
        'subsample_ratio': 0.8,         # Fraction of dataset sampled per split to estimate variance and CI
        'num_models': 10,               # Number of subsample splits evaluated
        'confidence_interval': 0.95,    # Confidence level for Student-t confidence interval
        'plot_mi_matrix': True,         # Generate and save mutual information matrix heatmap
    })

    def _run_evaluation(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
        factor_names: List[str],
        latent_dim_names: List[str],
    ) -> None:
        """Executes the MIG evaluation across multiple subsampled splits."""
        num_samples, num_latents = latent_array.shape
        num_factors = len(factor_names)

        num_bins = int(getattr(self, "num_bins", 20))
        num_models = max(1, int(getattr(self, "num_models", 10)))
        conf_level = float(getattr(self, "confidence_interval", 0.95))
        subsample_ratio = float(getattr(self, "subsample_ratio", 0.8))

        rng = np.random.default_rng()
        subsample_size = max(10, int(num_samples * subsample_ratio))

        all_mig_scores = []
        all_mi_matrices = []
        per_factor_migs = {fname: [] for fname in factor_names}
        per_factor_top_dims = {fname: [] for fname in factor_names}

        for _ in range(num_models):
            sub_idx = rng.choice(num_samples, size=subsample_size, replace=False)
            z_sub = latent_array[sub_idx]
            f_sub = {k: factor_dict[k][sub_idx] for k in factor_names}

            mi_matrix, factor_entropies = self._compute_mutual_info_matrix(
                latents=z_sub,
                factors_dict=f_sub,
                num_bins=num_bins,
            )
            all_mi_matrices.append(mi_matrix)

            factor_gaps = np.zeros(num_factors, dtype=float)

            for k_idx, fname in enumerate(factor_names):
                mi_k = mi_matrix[:, k_idx]
                sorted_indices = np.argsort(mi_k)[::-1]
                top_dim = sorted_indices[0]
                per_factor_top_dims[fname].append(top_dim)

                top_mi = mi_k[top_dim]
                second_mi = mi_k[sorted_indices[1]] if num_latents > 1 else 0.0
                h_k = factor_entropies[k_idx]

                if h_k > 1e-12:
                    gap = (top_mi - second_mi) / h_k
                else:
                    gap = 0.0

                gap_clipped = float(np.clip(gap, 0.0, 1.0))
                factor_gaps[k_idx] = gap_clipped
                per_factor_migs[fname].append(gap_clipped)

            all_mig_scores.append(float(np.mean(factor_gaps)))

        # Statistical aggregation
        mig_stats = self._compute_ci(all_mig_scores, confidence_level=conf_level)
        mean_mi_matrix = np.mean(all_mi_matrices, axis=0)
        conf_pct = int(conf_level * 100)

        # ---------------------------------------------------------------------
        # Log & Save Results
        # ---------------------------------------------------------------------
        self.log_result_msg("=" * 60)
        self.log_result_msg(
            f"DisentanglementMIGMetric (Overall Score): {mig_stats['mean']:.4f} ± {mig_stats['std']:.4f} "
            f"({conf_pct}% CI: [{mig_stats['ci_low']:.4f}, {mig_stats['ci_high']:.4f}])"
        )
        self.log_result_msg(f"Number of Evaluation Splits: {num_models} (Subsample size: {subsample_size})")
        self.log_result_msg(f"Histogram Discretization Bins: {num_bins}")
        self.log_result_msg("-" * 60)
        self.log_result_msg("Per-Factor Mutual Information Gap (MIG_k):")
        for fname in factor_names:
            gap_f = self._compute_ci(per_factor_migs[fname], confidence_level=conf_level)
            most_freq_dim_idx = int(np.bincount(per_factor_top_dims[fname]).argmax())
            most_freq_dim_name = latent_dim_names[most_freq_dim_idx]
            self.log_result_msg(
                f"  - Factor '{fname}': MIG = {gap_f['mean']:.4f} ± {gap_f['std']:.4f} "
                f"({conf_pct}% CI: [{gap_f['ci_low']:.4f}, {gap_f['ci_high']:.4f}]) | Primary Latent: '{most_freq_dim_name}'"
            )
        self.log_result_msg("=" * 60)

        # Save data files in data/
        np.save(os.path.join(self.data_dir, "mig_scores.npy"), np.array(all_mig_scores))
        np.save(os.path.join(self.data_dir, "mi_matrix.npy"), mean_mi_matrix)

        # Register metrics into metrics dictionary
        self.set_metric("mig_score", mig_stats["mean"])
        self.set_metric("mig", mig_stats["mean"])
        self.set_metric("mig_mean", mig_stats["mean"])
        self.set_metric("disentanglement_score", mig_stats["mean"])
        self.set_metric("mig_score_std", mig_stats["std"])

        # WandB logging
        if self.logger_type == "WandbLogger" and self.logger is not None:
            try:
                self.logger.experiment.log({
                    "[MIG] score_mean": mig_stats["mean"],
                    "[MIG] score_std": mig_stats["std"],
                })
            except Exception as e:
                self.log_warn(f"Failed to log MIG metrics to WandbLogger: {e}")

        # Heatmap plot
        if getattr(self, "plot_mi_matrix", True):
            self._plot_and_save_matrix_heatmap(
                matrix=mean_mi_matrix,
                row_names=latent_dim_names,
                col_names=factor_names,
                xlabel="Generative Factor",
                ylabel="Latent Dimension",
                title=f"Mutual Information Matrix I(z_i; v_k)\n(Overall MIG: {mig_stats['mean']:.3f} ± {mig_stats['ci']:.3f} 95% CI)",
                image_name="mutual_information_matrix",
                cbar_label="Mutual Information (nats)",
                cmap="Blues",
                val_format="{:.3f}",
            )
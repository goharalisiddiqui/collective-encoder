import os
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from collective_encoder.testplotters.disentanglement_metrics.base import BaseDisentanglementMetric


class DisentanglementSAPMetric(BaseDisentanglementMetric):
    r"""
    This implements the Separated Attribute Predictability (SAP) disentanglement metric.
    Theoretical basis: Kumar et al., ICLR 2018:
    "Variational Inference of Disentangled Latent Concepts from Unlabeled Observations" (DIP-VAE)
    https://openreview.net/pdf/4d42bf4c791265f2dc14a70b0ee3592e3bb6285d.pdf

    Key Algorithm:
    1. For each latent dimension z_i (i = 1..D) and each ground-truth generative factor v_k (k = 1..K):
       - Fit a univariate linear model using ONLY single dimension z_i to predict factor v_k.
       - Compute the predictability score S_{i, k} (e.g. test R^2 score clipped to [0, 1]) on a held-out test split.
    2. For each factor v_k, sort the predictability scores across all latent dimensions:
       S_{j_1, k} >= S_{j_2, k} >= ... >= S_{j_D, k}.
    3. Compute the SAP score for factor v_k:
       SAP_k = S_{j_1, k} - S_{j_2, k}.
    4. Overall SAP Score:
       SAP = (1 / K) \sum_{k=1}^K SAP_k \in [0, 1].
    5. Multi-model robustness: Evaluate over `num_models` random train/test splits with Student-t confidence intervals.
    """
    _IDENTIFIER = "DisentanglementSAPMetric"
    _OPTIONAL_ARGS = BaseDisentanglementMetric._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'test_split': 0.3,                 # Fraction of dataset held out for univariate predictability evaluation
        'num_models': 10,                  # Number of random train/test splits evaluated
        'confidence_interval': 0.95,       # Confidence level for Student-t confidence interval
        'plot_sap_matrix': True,           # Generate and save predictability score matrix heatmap
    })

    def _run_evaluation(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
        factor_names: List[str],
        latent_dim_names: List[str],
    ) -> None:
        """Executes the SAP metric evaluation across multiple random train/test splits."""
        num_latents = latent_array.shape[1]
        num_factors = len(factor_names)
        factor_matrix = np.column_stack([factor_dict[k] for k in factor_names])

        num_models = max(1, int(getattr(self, "num_models", 10)))
        conf_level = float(getattr(self, "confidence_interval", 0.95))
        test_split_ratio = float(getattr(self, "test_split", 0.3))

        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31 - 1, size=num_models)

        all_sap_scores = []
        all_score_matrices = []
        per_factor_saps = {fname: [] for fname in factor_names}
        per_factor_top_dims = {fname: [] for fname in factor_names}

        for seed_val in seeds:
            z_tr, z_te, y_tr, y_te = train_test_split(
                latent_array,
                factor_matrix,
                test_size=test_split_ratio,
                random_state=int(seed_val),
            )

            # Score matrix S: (num_latents, num_factors)
            S = np.zeros((num_latents, num_factors), dtype=float)

            for d_idx in range(num_latents):
                z_tr_d = z_tr[:, d_idx:d_idx+1]
                z_te_d = z_te[:, d_idx:d_idx+1]

                for k_idx in range(num_factors):
                    y_tr_k = y_tr[:, k_idx]
                    y_te_k = y_te[:, k_idx]

                    reg = LinearRegression()
                    reg.fit(z_tr_d, y_tr_k)

                    y_pred = reg.predict(z_te_d)
                    r2 = float(r2_score(y_te_k, y_pred))
                    # Predictability score: R^2 clipped to [0, 1]
                    S[d_idx, k_idx] = float(np.clip(r2, 0.0, 1.0))

            all_score_matrices.append(S)

            factor_saps = np.zeros(num_factors, dtype=float)

            for k_idx, fname in enumerate(factor_names):
                s_k = S[:, k_idx]
                sorted_indices = np.argsort(s_k)[::-1]
                top_dim = sorted_indices[0]
                per_factor_top_dims[fname].append(top_dim)

                top_score = s_k[top_dim]
                second_score = s_k[sorted_indices[1]] if num_latents > 1 else 0.0

                gap = float(np.clip(top_score - second_score, 0.0, 1.0))
                factor_saps[k_idx] = gap
                per_factor_saps[fname].append(gap)

            all_sap_scores.append(float(np.mean(factor_saps)))

        # Statistical aggregation
        sap_stats = self._compute_ci(all_sap_scores, confidence_level=conf_level)
        mean_score_matrix = np.mean(all_score_matrices, axis=0)
        conf_pct = int(conf_level * 100)

        # ---------------------------------------------------------------------
        # Log & Save Results
        # ---------------------------------------------------------------------
        self.log_result_msg("=" * 60)
        self.log_result_msg(
            f"DisentanglementSAPMetric (Overall SAP Score): {sap_stats['mean']:.4f} ± {sap_stats['std']:.4f} "
            f"({conf_pct}% CI: [{sap_stats['ci_low']:.4f}, {sap_stats['ci_high']:.4f}])"
        )
        self.log_result_msg(f"Number of Evaluation Splits: {num_models} (Test split: {test_split_ratio})")
        self.log_result_msg("-" * 60)
        self.log_result_msg("Per-Factor Separated Attribute Predictability (SAP_k):")
        for fname in factor_names:
            gap_f = self._compute_ci(per_factor_saps[fname], confidence_level=conf_level)
            most_freq_dim_idx = int(np.bincount(per_factor_top_dims[fname]).argmax())
            most_freq_dim_name = latent_dim_names[most_freq_dim_idx]
            self.log_result_msg(
                f"  - Factor '{fname}': SAP = {gap_f['mean']:.4f} ± {gap_f['std']:.4f} "
                f"({conf_pct}% CI: [{gap_f['ci_low']:.4f}, {gap_f['ci_high']:.4f}]) | Primary Latent: '{most_freq_dim_name}'"
            )
        self.log_result_msg("=" * 60)

        # Save data files in data/
        np.save(os.path.join(self.data_dir, "sap_scores.npy"), np.array(all_sap_scores))
        np.save(os.path.join(self.data_dir, "score_matrix.npy"), mean_score_matrix)

        # WandB logging
        if self.logger_type == "WandbLogger" and self.logger is not None:
            try:
                self.logger.experiment.log({
                    "[SAP] score_mean": sap_stats["mean"],
                    "[SAP] score_std": sap_stats["std"],
                })
            except Exception as e:
                self.log_warn(f"Failed to log SAP metrics to WandbLogger: {e}")

        # Heatmap plot
        if getattr(self, "plot_sap_matrix", True):
            self._plot_and_save_matrix_heatmap(
                matrix=mean_score_matrix,
                row_names=latent_dim_names,
                col_names=factor_names,
                xlabel="Generative Factor",
                ylabel="Latent Dimension",
                title=f"Univariate Predictability Score Matrix S(z_i; v_k) [R²]\n(Overall SAP: {sap_stats['mean']:.3f} ± {sap_stats['ci']:.3f} 95% CI)",
                image_name="sap_score_matrix",
                cbar_label="Univariate R² Score",
                cmap="Greens",
                val_format="{:.3f}",
            )
import os
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from collective_encoder.testplotters.disentanglement_metrics.base import BaseDisentanglementMetric


class DisentanglementModularityMetric(BaseDisentanglementMetric):
    r"""
    This implements the Modularity and Explicitness disentanglement metrics.
    Theoretical basis: Ridgeway and Mozer, NeurIPS 2018:
    "Learning Deep Disentangled Embeddings With the F-Statistic Loss"
    https://proceedings.neurips.cc/paper_files/paper/2018/file/2b24d495052a8ce66358eb576b8912c8-Paper.pdf

    Key Algorithm:
    1. Compute the empirical mutual information matrix I(z_i; v_k) between each latent dimension
       z_i (i = 1..D) and each ground-truth generative factor v_k (k = 1..K) using discrete binning.
    2. Modularity per Latent Dimension:
       - For dimension i, find the factor with maximum mutual information:
         k_i* = argmax_k I(z_i; v_k), with peak value \mu_i* = I(z_i; v_{k_i*}).
       - Compute the modularity score for dimension i:
         M_i = \sum_{k \ne k_i*} (1 - I(z_i; v_k) / \mu_i*)^2 / (K - 1)  (if \mu_i* > 0, else 0).
    3. Overall Modularity Score:
       M = (1 / D) \sum_{i=1}^D M_i \in [0, 1].
    4. Explicitness:
       Average predictability (test R^2 / accuracy) when decoding each factor from latents using linear models.
    5. Multi-split robustness: Subsample data over `num_models` random splits to report Mean ± Std & CI.
    """
    _IDENTIFIER = "DisentanglementModularityMetric"
    _OPTIONAL_ARGS = BaseDisentanglementMetric._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'num_bins': 20,                 # Number of histogram bins for discrete mutual information estimation
        'subsample_ratio': 0.8,         # Fraction of dataset sampled per split
        'num_models': 10,               # Number of subsample splits evaluated
        'confidence_interval': 0.95,    # Confidence level for Student-t confidence interval
        'compute_explicitness': True,   # Compute and log Explicitness score alongside Modularity
        'plot_mi_matrix': True,         # Generate and save mutual information matrix heatmap
    })

    def _run_evaluation(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
        factor_names: List[str],
        latent_dim_names: List[str],
    ) -> None:
        """Executes the Modularity evaluation across multiple subsampled splits."""
        num_samples, num_latents = latent_array.shape
        num_factors = len(factor_names)
        factor_matrix = np.column_stack([factor_dict[k] for k in factor_names])

        num_bins = int(getattr(self, "num_bins", 20))
        num_models = max(1, int(getattr(self, "num_models", 10)))
        conf_level = float(getattr(self, "confidence_interval", 0.95))
        subsample_ratio = float(getattr(self, "subsample_ratio", 0.8))
        eval_explicitness = bool(getattr(self, "compute_explicitness", True))

        rng = np.random.default_rng()
        subsample_size = max(10, int(num_samples * subsample_ratio))

        all_modularity_scores = []
        all_explicitness_scores = []
        all_mi_matrices = []
        per_dim_modularities = {dname: [] for dname in latent_dim_names}
        per_dim_primary_factors = {dname: [] for dname in latent_dim_names}

        for _ in range(num_models):
            sub_idx = rng.choice(num_samples, size=subsample_size, replace=False)
            z_sub = latent_array[sub_idx]
            f_sub = {k: factor_dict[k][sub_idx] for k in factor_names}

            mi_matrix, _ = self._compute_mutual_info_matrix(
                latents=z_sub,
                factors_dict=f_sub,
                num_bins=num_bins,
            )
            all_mi_matrices.append(mi_matrix)

            dim_modularities = np.zeros(num_latents, dtype=float)

            for d_idx, dname in enumerate(latent_dim_names):
                mi_d = mi_matrix[d_idx, :]
                k_star = int(np.argmax(mi_d))
                mu_star = mi_d[k_star]
                per_dim_primary_factors[dname].append(k_star)

                if mu_star > 1e-12 and num_factors > 1:
                    other_factors = [k for k in range(num_factors) if k != k_star]
                    squared_devs = (1.0 - mi_d[other_factors] / mu_star) ** 2
                    mod_d = float(np.sum(squared_devs) / (num_factors - 1))
                else:
                    mod_d = 0.0

                mod_d_clipped = float(np.clip(mod_d, 0.0, 1.0))
                dim_modularities[d_idx] = mod_d_clipped
                per_dim_modularities[dname].append(mod_d_clipped)

            all_modularity_scores.append(float(np.mean(dim_modularities)))

            # Explicitness computation (Linear decoding performance)
            if eval_explicitness:
                z_tr, z_te, y_tr, y_te = train_test_split(
                    z_sub,
                    factor_matrix[sub_idx],
                    test_size=0.3,
                    random_state=rng.integers(0, 2**31 - 1),
                )
                r2_scores = []
                for k in range(num_factors):
                    reg = Ridge(alpha=1.0)
                    reg.fit(z_tr, y_tr[:, k])
                    y_pred = reg.predict(z_te)
                    r2_scores.append(max(0.0, float(r2_score(y_te[:, k], y_pred))))
                all_explicitness_scores.append(float(np.mean(r2_scores)))

        # Statistical aggregation
        mod_stats = self._compute_ci(all_modularity_scores, confidence_level=conf_level)
        mean_mi_matrix = np.mean(all_mi_matrices, axis=0)
        conf_pct = int(conf_level * 100)

        # ---------------------------------------------------------------------
        # Log & Save Results
        # ---------------------------------------------------------------------
        self.log_result_msg("=" * 60)
        self.log_result_msg(
            f"DisentanglementModularityMetric (Modularity): {mod_stats['mean']:.4f} ± {mod_stats['std']:.4f} "
            f"({conf_pct}% CI: [{mod_stats['ci_low']:.4f}, {mod_stats['ci_high']:.4f}])"
        )
        if eval_explicitness:
            exp_stats = self._compute_ci(all_explicitness_scores, confidence_level=conf_level)
            self.log_result_msg(
                f"Explicitness (Linear R² Predictability):    {exp_stats['mean']:.4f} ± {exp_stats['std']:.4f} "
                f"({conf_pct}% CI: [{exp_stats['ci_low']:.4f}, {exp_stats['ci_high']:.4f}])"
            )
        self.log_result_msg(f"Number of Evaluation Splits: {num_models} (Subsample size: {subsample_size})")
        self.log_result_msg(f"Histogram Discretization Bins: {num_bins}")
        self.log_result_msg("-" * 60)
        self.log_result_msg("Per-Dimension Modularity (M_i):")
        for dname in latent_dim_names:
            m_dim = self._compute_ci(per_dim_modularities[dname], confidence_level=conf_level)
            most_freq_k = int(np.bincount(per_dim_primary_factors[dname]).argmax())
            primary_fname = factor_names[most_freq_k]
            self.log_result_msg(
                f"  - Dimension '{dname}': Modularity = {m_dim['mean']:.4f} ± {m_dim['std']:.4f} "
                f"({conf_pct}% CI: [{m_dim['ci_low']:.4f}, {m_dim['ci_high']:.4f}]) | Primary Factor: '{primary_fname}'"
            )
        self.log_result_msg("=" * 60)

        # Save data files in data/
        np.save(os.path.join(self.data_dir, "modularity_scores.npy"), np.array(all_modularity_scores))
        if eval_explicitness:
            np.save(os.path.join(self.data_dir, "explicitness_scores.npy"), np.array(all_explicitness_scores))
        np.save(os.path.join(self.data_dir, "mi_matrix.npy"), mean_mi_matrix)

        # WandB logging
        if self.logger_type == "WandbLogger" and self.logger is not None:
            try:
                log_dict = {
                    "[Modularity] score_mean": mod_stats["mean"],
                    "[Modularity] score_std": mod_stats["std"],
                }
                if eval_explicitness:
                    log_dict["[Modularity] explicitness_mean"] = exp_stats["mean"]
                self.logger.experiment.log(log_dict)
            except Exception as e:
                self.log_warn(f"Failed to log Modularity metrics to WandbLogger: {e}")

        # Heatmap plot
        if getattr(self, "plot_mi_matrix", True):
            self._plot_and_save_matrix_heatmap(
                matrix=mean_mi_matrix,
                row_names=latent_dim_names,
                col_names=factor_names,
                xlabel="Generative Factor",
                ylabel="Latent Dimension",
                title=f"Mutual Information Matrix I(z_i; v_k)\n(Overall Modularity: {mod_stats['mean']:.3f} ± {mod_stats['ci']:.3f} 95% CI)",
                image_name="modularity_matrix",
                cbar_label="Mutual Information (nats)",
                cmap="PuBu",
                val_format="{:.3f}",
            )
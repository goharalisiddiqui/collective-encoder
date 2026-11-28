import os
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from collective_encoder.testplotters.disentanglement_metrics.base import BaseDisentanglementMetric


class DisentanglementDCIMetric(BaseDisentanglementMetric):
    r"""
    This implements the Disentangled Representation Complexity (DCI) disentanglement metric.
    Theoretical basis: Eastwood and Williams, ICLR 2018:
    "A Framework for the Quantitative Evaluation of Disentangled Representations"
    https://openreview.net/pdf?id=By-7dz-AZ

    Key Algorithm:
    1. For each ground-truth generative factor v_k (k = 1..K), train a predictive model
       (default: GradientBoostingRegressor) to predict v_k using all latent variables z \in R^D.
    2. Extract the feature importance matrix R \in R^{D x K}, where R_{i, k} >= 0 represents
       the importance of latent dimension z_i in predicting generative factor v_k.
    3. Disentanglement (D):
       - Normalize dimension importances across factors: P_{i, k} = R_{i, k} / \sum_{k'} R_{i, k'}.
       - Compute dimension entropy: H(P_i) = - \sum_{k} P_{i, k} \log_K(P_{i, k}).
       - Dimension disentanglement: D_i = 1 - H(P_i).
       - Overall Disentanglement: D = \sum_i \rho_i D_i, where \rho_i = \sum_k R_{i, k} / \sum_{i', k} R_{i', k}.
    4. Completeness (C):
       - Normalize factor importances across latents: \tilde{P}_{i, k} = R_{i, k} / \sum_{i'} R_{i', k}.
       - Factor completeness: C_k = 1 - H(\tilde{P}_k), where H(\tilde{P}_k) = - \sum_i \tilde{P}_{i, k} \log_D(\tilde{P}_{i, k}).
       - Overall Completeness: C = (1 / K) \sum_k C_k.
    5. Informativeness (I):
       - Average test prediction accuracy / R^2 score across all K factors evaluated on held-out test splits.
    6. Multi-model robustness: Evaluate over `num_models` random train/test splits with Student-t confidence intervals.
    """
    _IDENTIFIER = "DisentanglementDCIMetric"
    _OPTIONAL_ARGS = BaseDisentanglementMetric._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'regressor_type': 'gradient_boosting',  # 'gradient_boosting', 'random_forest', 'ridge', 'lasso'
        'regressor_kwargs': None,
        'test_split': 0.3,
        'num_models': 10,
        'confidence_interval': 0.95,
        'plot_importance_matrix': True,
    })

    def _get_regressor(self, seed: Optional[int] = None) -> Any:
        """Instantiates the regressor used to estimate feature importances."""
        kwargs = self.regressor_kwargs.copy() if isinstance(self.regressor_kwargs, dict) else {}
        rtype = str(self.regressor_type).lower()

        if "random_state" not in kwargs and seed is not None:
            kwargs["random_state"] = int(seed)

        if rtype == "gradient_boosting":
            if "n_estimators" not in kwargs:
                kwargs["n_estimators"] = 100
            return GradientBoostingRegressor(**kwargs)
        elif rtype == "random_forest":
            if "n_estimators" not in kwargs:
                kwargs["n_estimators"] = 100
            return RandomForestRegressor(**kwargs)
        elif rtype == "ridge":
            return Ridge(**kwargs)
        elif rtype == "lasso":
            return Lasso(**kwargs)
        else:
            self.raise_error(
                f"Unknown regressor_type '{self.regressor_type}'. "
                f"Supported types: 'gradient_boosting', 'random_forest', 'ridge', 'lasso'."
            )

    def _run_evaluation(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
        factor_names: List[str],
        latent_dim_names: List[str],
    ) -> None:
        """Executes the DCI metric evaluation across multiple random train/test splits."""
        num_latents = latent_array.shape[1]
        num_factors = len(factor_names)
        factor_matrix = np.column_stack([factor_dict[k] for k in factor_names])

        num_models = max(1, int(getattr(self, "num_models", 10)))
        conf_level = float(getattr(self, "confidence_interval", 0.95))
        test_split_ratio = float(getattr(self, "test_split", 0.3))

        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31 - 1, size=num_models)

        all_disentanglement = []
        all_completeness = []
        all_informativeness = []
        all_importance_matrices = []

        per_dim_disentanglement = {dim_name: [] for dim_name in latent_dim_names}
        per_factor_completeness = {fname: [] for fname in factor_names}
        per_factor_informativeness = {fname: [] for fname in factor_names}

        for seed_val in seeds:
            z_train, z_test, y_train, y_test = train_test_split(
                latent_array,
                factor_matrix,
                test_size=test_split_ratio,
                random_state=int(seed_val),
            )

            # R matrix: (num_latents, num_factors)
            R = np.zeros((num_latents, num_factors), dtype=float)
            test_scores = np.zeros(num_factors, dtype=float)

            for k_idx in range(num_factors):
                model_k = self._get_regressor(seed=seed_val)
                model_k.fit(z_train, y_train[:, k_idx])

                y_pred = model_k.predict(z_test)
                r2 = float(r2_score(y_test[:, k_idx], y_pred))
                # Clip negative R2 at 0
                test_scores[k_idx] = max(0.0, r2)
                per_factor_informativeness[factor_names[k_idx]].append(test_scores[k_idx])

                # Extract feature importance
                if hasattr(model_k, "feature_importances_"):
                    R[:, k_idx] = model_k.feature_importances_
                elif hasattr(model_k, "coef_"):
                    R[:, k_idx] = np.abs(model_k.coef_)
                else:
                    self.raise_error(f"Regressor {type(model_k).__name__} has neither feature_importances_ nor coef_.")

            all_importance_matrices.append(R)

            # -----------------------------------------------------------------
            # 1. Disentanglement Score (D)
            # -----------------------------------------------------------------
            R_sum_factors = np.sum(R, axis=1)  # (D,)
            R_total = np.sum(R)
            rho = R_sum_factors / (R_total + 1e-12)

            D_i = np.zeros(num_latents, dtype=float)
            log_K = np.log(num_factors) if num_factors > 1 else 1.0

            for i in range(num_latents):
                if R_sum_factors[i] > 1e-12:
                    p_i = R[i, :] / R_sum_factors[i]
                    p_nz = p_i[p_i > 0]
                    entropy_i = -np.sum(p_nz * np.log(p_nz)) / log_K
                    D_i[i] = max(0.0, 1.0 - entropy_i)
                else:
                    D_i[i] = 1.0  # Inactive dimension is disentangled
                per_dim_disentanglement[latent_dim_names[i]].append(D_i[i])

            D_overall = float(np.sum(rho * D_i))
            all_disentanglement.append(D_overall)

            # -----------------------------------------------------------------
            # 2. Completeness Score (C)
            # -----------------------------------------------------------------
            R_sum_latents = np.sum(R, axis=0)  # (K,)
            C_k = np.zeros(num_factors, dtype=float)
            log_D = np.log(num_latents) if num_latents > 1 else 1.0

            for k in range(num_factors):
                if R_sum_latents[k] > 1e-12:
                    p_k = R[:, k] / R_sum_latents[k]
                    p_nz = p_k[p_k > 0]
                    entropy_k = -np.sum(p_nz * np.log(p_nz)) / log_D
                    C_k[k] = max(0.0, 1.0 - entropy_k)
                else:
                    C_k[k] = 1.0
                per_factor_completeness[factor_names[k]].append(C_k[k])

            C_overall = float(np.mean(C_k))
            all_completeness.append(C_overall)

            # -----------------------------------------------------------------
            # 3. Informativeness Score (I)
            # -----------------------------------------------------------------
            I_overall = float(np.mean(test_scores))
            all_informativeness.append(I_overall)

        # Statistical aggregation
        d_stats = self._compute_ci(all_disentanglement, confidence_level=conf_level)
        c_stats = self._compute_ci(all_completeness, confidence_level=conf_level)
        i_stats = self._compute_ci(all_informativeness, confidence_level=conf_level)

        mean_R = np.mean(all_importance_matrices, axis=0)
        conf_pct = int(conf_level * 100)

        # ---------------------------------------------------------------------
        # Log & Save Results
        # ---------------------------------------------------------------------
        self.log_result_msg("=" * 60)
        self.log_result_msg(f"DisentanglementDCIMetric Summary ({num_models} models, {conf_pct}% CI):")
        self.log_result_msg(
            f"  - Disentanglement (D):  {d_stats['mean']:.4f} ± {d_stats['std']:.4f} "
            f"({conf_pct}% CI: [{d_stats['ci_low']:.4f}, {d_stats['ci_high']:.4f}])"
        )
        self.log_result_msg(
            f"  - Completeness (C):     {c_stats['mean']:.4f} ± {c_stats['std']:.4f} "
            f"({conf_pct}% CI: [{c_stats['ci_low']:.4f}, {c_stats['ci_high']:.4f}])"
        )
        self.log_result_msg(
            f"  - Informativeness (I):  {i_stats['mean']:.4f} ± {i_stats['std']:.4f} "
            f"({conf_pct}% CI: [{i_stats['ci_low']:.4f}, {i_stats['ci_high']:.4f}])"
        )
        self.log_result_msg("-" * 60)
        self.log_result_msg("Per-Factor Completeness & Informativeness (R²):")
        for fname in factor_names:
            c_f = self._compute_ci(per_factor_completeness[fname], confidence_level=conf_level)
            i_f = self._compute_ci(per_factor_informativeness[fname], confidence_level=conf_level)
            self.log_result_msg(
                f"  - Factor '{fname}': Completeness = {c_f['mean']:.4f} ± {c_f['std']:.4f} | "
                f"Informativeness (R²) = {i_f['mean']:.4f} ± {i_f['std']:.4f}"
            )
        self.log_result_msg("-" * 60)
        self.log_result_msg("Per-Dimension Disentanglement:")
        for dname in latent_dim_names:
            d_dim = self._compute_ci(per_dim_disentanglement[dname], confidence_level=conf_level)
            self.log_result_msg(f"  - Dimension '{dname}': Disentanglement = {d_dim['mean']:.4f} ± {d_dim['std']:.4f}")
        self.log_result_msg("=" * 60)

        # Save data files in data/
        dci_arr = np.column_stack([all_disentanglement, all_completeness, all_informativeness])
        np.save(os.path.join(self.data_dir, "dci_scores.npy"), dci_arr)
        np.save(os.path.join(self.data_dir, "importance_matrix.npy"), mean_R)

        # Register metrics into metrics dictionary
        self.set_metric("dci_disentanglement", d_stats["mean"])
        self.set_metric("dci_completeness", c_stats["mean"])
        self.set_metric("dci_informativeness", i_stats["mean"])
        self.set_metric("dci_d", d_stats["mean"])
        self.set_metric("dci_c", c_stats["mean"])
        self.set_metric("dci_i", i_stats["mean"])
        self.set_metric("disentanglement_score", d_stats["mean"])

        # WandB logging
        if self.logger_type == "WandbLogger" and self.logger is not None:
            try:
                self.logger.experiment.log({
                    "[DCI] disentanglement_mean": d_stats["mean"],
                    "[DCI] completeness_mean": c_stats["mean"],
                    "[DCI] informativeness_mean": i_stats["mean"],
                })
            except Exception as e:
                self.log_warn(f"Failed to log DCI metrics to WandbLogger: {e}")

        # Heatmap plot
        if getattr(self, "plot_importance_matrix", True):
            self._plot_and_save_matrix_heatmap(
                matrix=mean_R,
                row_names=latent_dim_names,
                col_names=factor_names,
                xlabel="Generative Factor",
                ylabel="Latent Dimension",
                title=f"DCI Feature Importance Matrix (R)\n(D: {d_stats['mean']:.3f} ± {d_stats['ci']:.3f}, C: {c_stats['mean']:.3f} ± {c_stats['ci']:.3f})",
                image_name="importance_matrix",
                cbar_label="Relative Feature Importance",
                cmap="YlGnBu",
                val_format="{:.3f}",
            )
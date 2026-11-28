import os
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from collective_encoder.testplotters.disentanglement_metrics.base import BaseDisentanglementMetric


class MajorityVoteClassifier:
    r"""
    Majority-vote classifier for discrete input features (e.g. latent dimension index).
    Maps each latent dimension d* \in {0, ..., D-1} to the most frequent ground-truth factor k.
    As proven in Kim & Mnih (ICML 2018), this is the optimal, deterministic, parameter-free
    classifier when both inputs and targets are discrete.
    """
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.vote_map_: Dict[int, int] = {}
        self.majority_class_: int = 0
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityVoteClassifier":
        X_flat = np.squeeze(np.asarray(X, dtype=int))
        y_flat = np.squeeze(np.asarray(y, dtype=int))
        self.classes_ = np.unique(y_flat)

        if len(y_flat) == 0:
            return self

        # Overall fallback majority class
        u_classes, counts = np.unique(y_flat, return_counts=True)
        self.majority_class_ = int(u_classes[np.argmax(counts)])

        # Majority vote per discrete feature / dimension
        self.vote_map_ = {}
        for d in np.unique(X_flat):
            mask = (X_flat == d)
            y_d = y_flat[mask]
            c_vals, c_counts = np.unique(y_d, return_counts=True)
            self.vote_map_[int(d)] = int(c_vals[np.argmax(c_counts)])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = np.squeeze(np.asarray(X, dtype=int))
        if X_flat.ndim == 0:
            return np.array([self.vote_map_.get(int(X_flat), self.majority_class_)])
        return np.array([self.vote_map_.get(int(d), self.majority_class_) for d in X_flat], dtype=int)


class DisentanglementFactorMetric(BaseDisentanglementMetric):
    """
    This implements the FactorVAE test metric for disentangled representations.
    Theoretical basis: Kim and Mnih et al. (ICML 2018):
    "Disentangling by Factorising"
    https://proceedings.mlr.press/v80/kim18b/kim18b.pdf

    Key Algorithm:
    1. Globally normalize all latent dimensions by their empirical standard deviation:
       z_hat_d = z_d / std(z_d)
    2. For each factor k, generate batches of size L where factor k is held constant
       while other factors vary.
    3. Compute empirical variance V_d for each normalized latent dimension in the batch.
    4. The feature is the dimension with minimum variance: d* = argmin_d V_d.
       The target class is the index of the fixed generative factor k.
    5. Evaluate classification accuracy using a Majority-Vote Classifier across held-out splits.
    """
    _IDENTIFIER = "DisentanglementFactorMetric"
    _OPTIONAL_ARGS = BaseDisentanglementMetric._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'batch_size_L': 50,                 # Number of samples per batch to compute empirical variance (L >= 2)
        'max_batches_per_factor': 1000,     # Maximum variance batches generated per factor category
    })

    def _log_and_save_results(self, results: Dict[str, Any], factor_names: List[str]) -> None:
        super()._log_and_save_results(results, factor_names)
        t_mean = results["test_accuracy_stats"]["mean"]
        tr_mean = results["train_accuracy_stats"]["mean"]
        self.set_metric("factor_score", t_mean)
        self.set_metric("factor_vae_score", t_mean)
        self.set_metric("factor_train_acc", tr_mean)

    def _construct_dataset(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, Any]]:
        """
        1. Normalizes latent dimensions by empirical standard deviation.
        2. Gathers batches of size L for each fixed factor and calculates per-dimension variance.
        3. Identifies d* = argmin_d Var(z_hat_d) as feature X, and factor index k as label y.
        """
        rng = np.random.default_rng()
        factor_names = list(factor_dict.keys())
        num_samples, num_latents = latent_array.shape

        # Step 1: Global representation normalization by empirical standard deviation
        global_std = np.std(latent_array, axis=0, ddof=1 if num_samples > 1 else 0)
        # Avoid division by zero for inactive/constant dimensions
        global_std = np.where(global_std > 1e-12, global_std, 1.0)
        norm_latents = latent_array / global_std

        factor_matrix = np.column_stack([factor_dict[k] for k in factor_names])
        num_factors = len(factor_names)

        l_batch = max(2, int(getattr(self, "batch_size_L", 50)))
        max_batches = int(getattr(self, "max_batches_per_factor", 1000))

        all_argmin_dims = []
        all_y = []
        factor_sample_counts = {}
        diagnostic_info = {
            "factor_batches": {},
            "variance_profiles": {},
            "norm_latents": norm_latents,
            "global_std": global_std,
        }

        for k_idx, factor_name in enumerate(factor_names):
            tol_k = self._get_factor_tolerance(factor_name)
            var_thresh_k = self._get_variation_threshold(factor_name)

            target_vals = factor_matrix[:, k_idx]
            other_indices = [idx for idx in range(num_factors) if idx != k_idx]
            other_vals = factor_matrix[:, other_indices]

            # Sort samples by target factor for fast binary range search
            sort_order = np.argsort(target_vals)
            sorted_targets = target_vals[sort_order]
            sorted_others = other_vals[sort_order]
            sorted_latents = norm_latents[sort_order]

            batches_indices = []
            batches_variances = []
            argmin_dims = []

            # Randomize anchor search order to sample diverse factor values
            anchor_indices = rng.permutation(num_samples)

            for i in anchor_indices:
                val_i = sorted_targets[i]
                left = np.searchsorted(sorted_targets, val_i - tol_k, side='left')
                right = np.searchsorted(sorted_targets, val_i + tol_k, side='right')

                if right - left < 2:
                    continue

                candidates = np.arange(left, right)
                diff_others = np.abs(sorted_others[candidates] - sorted_others[i])
                max_diff = np.max(diff_others, axis=-1)
                valid = candidates[max_diff >= var_thresh_k]

                # We need valid points that vary in other factors plus anchor point
                valid_set = np.unique(np.append(valid, i))
                if len(valid_set) < 2:
                    continue

                # Sample batch of size L (or all valid points if fewer than L)
                if len(valid_set) >= l_batch:
                    batch_idx = rng.choice(valid_set, size=l_batch, replace=False)
                else:
                    batch_idx = valid_set

                # Compute empirical variance across batch for each normalized latent dimension
                batch_latents = sorted_latents[batch_idx]
                batch_var = np.var(batch_latents, axis=0, ddof=1 if len(batch_idx) > 1 else 0)

                # Feature: index of dimension with lowest variance
                min_dim = int(np.argmin(batch_var))

                batches_indices.append(sort_order[batch_idx])
                batches_variances.append(batch_var)
                argmin_dims.append(min_dim)

                if len(argmin_dims) >= max_batches:
                    break

            count = len(argmin_dims)
            factor_sample_counts[factor_name] = count

            if count == 0:
                self.log_warn(
                    f"Category '{factor_name}' has 0 valid variance batches "
                    f"(tolerance={tol_k}, variation_threshold={var_thresh_k}, batch_size_L={l_batch}). Skipping category."
                )
                continue

            if count < self.min_samples_warning:
                self.log_warn(
                    f"Category '{factor_name}' has only {count} variance batches "
                    f"(< {self.min_samples_warning}). "
                    f"The test data may be too small for the metric to be robust."
                )

            all_argmin_dims.extend(argmin_dims)
            all_y.extend([k_idx] * count)

            diagnostic_info["factor_batches"][factor_name] = batches_indices
            diagnostic_info["variance_profiles"][factor_name] = np.array(batches_variances)

        if len(all_argmin_dims) == 0:
            return np.empty((0, 1), dtype=int), np.empty(0, dtype=int), factor_sample_counts, diagnostic_info

        X = np.array(all_argmin_dims, dtype=int)[:, np.newaxis]
        y = np.array(all_y, dtype=int)

        return X, y, factor_sample_counts, diagnostic_info

    def _get_classifier(self, seed: Optional[int] = None) -> MajorityVoteClassifier:
        """Instantiates the optimal, deterministic Majority-Vote Classifier."""
        return MajorityVoteClassifier(random_state=seed)

    def _plot_and_save_training_points(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
        diagnostic_info: Any,
        X: np.ndarray,
        y: np.ndarray,
        factor_names: List[str],
        latent_dim_names: Optional[List[str]] = None,
    ) -> None:
        """
        Plots the FactorVAE metric training data:
        1. In the generative factor space showing batch clusters for each factor.
        2. Normalized latent variance profiles showing which latent dimension exhibits minimum variance.
        """
        if not isinstance(diagnostic_info, dict):
            return

        factor_batches = diagnostic_info.get("factor_batches", {})
        variance_profiles = diagnostic_info.get("variance_profiles", {})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(factor_names))))

        # Left panel: Generative factor space clusters
        f_x_name = factor_names[0]
        f_y_name = factor_names[1] if len(factor_names) > 1 else factor_names[0]
        f_x = factor_dict[f_x_name]
        f_y = factor_dict[f_y_name]

        bg_subsample = min(5000, len(f_x))
        bg_idx = np.random.choice(len(f_x), size=bg_subsample, replace=False)
        ax1.scatter(f_x[bg_idx], f_y[bg_idx], c='lightgray', s=8, alpha=0.3, label='Dataset Samples')

        for k_idx, fname in enumerate(factor_names):
            batches = factor_batches.get(fname, [])
            if len(batches) == 0:
                continue

            num_disp_batches = min(50, len(batches))
            disp_batch_idx = np.random.choice(len(batches), size=num_disp_batches, replace=False)
            sampled_pts = np.unique(np.concatenate([batches[b] for b in disp_batch_idx]))

            color = colors[k_idx % len(colors)]
            ax1.scatter(f_x[sampled_pts], f_y[sampled_pts], color=color, s=16, alpha=0.7, label=f"Fixed '{fname}' batches")

        ax1.set_xlabel(f"Generative Factor: {f_x_name}", fontsize=10)
        ax1.set_ylabel(f"Generative Factor: {f_y_name}", fontsize=10)
        ax1.set_title("Constant Factor Batches in Generative Space", fontsize=11)
        ax1.legend(loc='best', fontsize=8)

        # Right panel: Mean normalized variance per latent dimension
        num_latents = latent_array.shape[1]
        ld_labels = latent_dim_names if latent_dim_names else [f"LD_{i+1}" for i in range(num_latents)]
        x_positions = np.arange(num_latents)
        bar_width = 0.8 / max(1, len(factor_names))

        for k_idx, fname in enumerate(factor_names):
            v_prof = variance_profiles.get(fname)
            if v_prof is None or len(v_prof) == 0:
                continue

            mean_var = np.mean(v_prof, axis=0)
            color = colors[k_idx % len(colors)]
            ax2.bar(
                x_positions + (k_idx - len(factor_names) / 2.0 + 0.5) * bar_width,
                mean_var,
                width=bar_width,
                color=color,
                alpha=0.8,
                label=f"Fixed '{fname}'",
            )

        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(ld_labels, fontsize=9, rotation=30, ha="right")
        ax2.set_xlabel("Normalized Latent Dimension", fontsize=10)
        ax2.set_ylabel("Empirical Batch Variance Var(z_hat)", fontsize=10)
        ax2.set_title("Mean Batch Variance per Latent Dimension (argmin selected)", fontsize=11)
        ax2.legend(loc='best', fontsize=8)

        plt.tight_layout()
        self.log_image(fig, "training_points", subpath="data")
        plt.close(fig)
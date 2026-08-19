from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from collective_encoder.testplotters.disentanglement_metrics.base import BaseDisentanglementMetric


class DisentanglementBetaMetric(BaseDisentanglementMetric):
    """
    This implements the Beta-VAE test metric for disentangled representations.
    Theoretical basis: Higgins et al. (ICLR 2017):
    "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
    https://papers.baulab.info/papers/also/Higgins-2017.pdf

    It evaluates disentanglement by constructing difference vectors |z_1 - z_2| for sample
    pairs where a specific ground-truth factor is kept constant while other factors vary,
    and training a linear classifier to identify which generative factor was held constant.
    """
    _IDENTIFIER = "DisentanglementBetaMetric"
    _OPTIONAL_ARGS = BaseDisentanglementMetric._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'batch_size_L': 1,
        'max_pairs_per_factor': 10000,
        'classifier_type': 'logistic_regression',
        'classifier_kwargs': None,
    })

    def _log_and_save_results(self, results: Dict[str, Any], factor_names: List[str]) -> None:
        super()._log_and_save_results(results, factor_names)
        t_mean = results["test_accuracy_stats"]["mean"]
        tr_mean = results["train_accuracy_stats"]["mean"]
        self.set_metric("beta_score", t_mean)
        self.set_metric("beta_vae_score", t_mean)
        self.set_metric("beta_train_acc", tr_mean)

    def _construct_dataset(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Finds pairs of data points for each factor where that factor is held constant
        within tolerance while other factors vary, and calculates difference vectors |z_1 - z_2|.
        Uses sorted binary search for fast O(N log N) matching without O(N^2) memory overhead.
        """
        rng = np.random.default_rng()
        factor_names = list(factor_dict.keys())
        num_samples = latent_array.shape[0]

        all_z_diff = []
        all_y = []
        factor_sample_counts = {}
        factor_pairs = {}

        factor_matrix = np.column_stack([factor_dict[k] for k in factor_names])
        num_factors = len(factor_names)
        max_pairs = int(getattr(self, "max_pairs_per_factor", 10000))

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
            sorted_latents = latent_array[sort_order]

            i_pairs = []
            j_pairs = []

            # Randomize anchor evaluation order to avoid sampling bias across dataset
            anchor_indices = rng.permutation(num_samples)

            for i in anchor_indices:
                val_i = sorted_targets[i]
                # Find range of indices with |target_val - val_i| <= tol_k
                left = np.searchsorted(sorted_targets, val_i - tol_k, side='left')
                right = np.searchsorted(sorted_targets, val_i + tol_k, side='right')

                if right - left <= 1:
                    continue

                candidates = np.arange(left, right)
                candidates = candidates[candidates != i]

                if len(candidates) == 0:
                    continue

                # Check variation in other factors
                diff_others = np.abs(sorted_others[candidates] - sorted_others[i])
                max_diff = np.max(diff_others, axis=-1)
                valid = candidates[max_diff >= var_thresh_k]

                for v in valid:
                    # Enforce ordering to avoid duplicate mirror pairs
                    orig_i, orig_v = sort_order[i], sort_order[v]
                    if orig_i < orig_v:
                        i_pairs.append(i)
                        j_pairs.append(v)
                        if len(i_pairs) >= max_pairs:
                            break

                if len(i_pairs) >= max_pairs:
                    break

            num_pairs = len(i_pairs)
            if num_pairs == 0:
                self.log_warn(
                    f"Category '{factor_name}' has 0 valid pair data points "
                    f"(tolerance={tol_k}, variation_threshold={var_thresh_k}). Skipping category."
                )
                factor_sample_counts[factor_name] = 0
                continue

            i_arr = np.array(i_pairs)
            j_arr = np.array(j_pairs)
            factor_pairs[factor_name] = (sort_order[i_arr], sort_order[j_arr])
            pair_diffs = np.abs(sorted_latents[i_arr] - sorted_latents[j_arr])

            l_batch = max(1, int(self.batch_size_L))
            if l_batch == 1:
                diff_vectors = pair_diffs
            else:
                perm = rng.permutation(num_pairs)
                num_batches = num_pairs // l_batch
                if num_batches == 0:
                    diff_vectors = np.mean(pair_diffs, axis=0, keepdims=True)
                else:
                    trimmed_indices = perm[:num_batches * l_batch].reshape(num_batches, l_batch)
                    diff_vectors = np.mean(pair_diffs[trimmed_indices], axis=1)

            count = diff_vectors.shape[0]
            factor_sample_counts[factor_name] = count

            if count < self.min_samples_warning:
                self.log_warn(
                    f"Category '{factor_name}' has only {count} data points "
                    f"(< {self.min_samples_warning}). "
                    f"The test data is too small for the metric to be useful."
                )

            all_z_diff.append(diff_vectors)
            all_y.append(np.full(count, k_idx, dtype=int))

        if len(all_z_diff) == 0:
            return np.empty((0, latent_array.shape[1])), np.empty(0, dtype=int), factor_sample_counts, factor_pairs

        z_diff_array = np.vstack(all_z_diff)
        y_diff_array = np.concatenate(all_y)

        return z_diff_array, y_diff_array, factor_sample_counts, factor_pairs

    def _get_classifier(self, seed: Optional[int] = None):
        kwargs = self.classifier_kwargs.copy() if isinstance(self.classifier_kwargs, dict) else {}
        ctype = str(self.classifier_type).lower()

        if "random_state" not in kwargs and seed is not None:
            kwargs["random_state"] = int(seed)

        if ctype == "logistic_regression":
            if "max_iter" not in kwargs:
                kwargs["max_iter"] = 1000
            return LogisticRegression(**kwargs)
        elif ctype == "linear_svc":
            if "max_iter" not in kwargs:
                kwargs["max_iter"] = 2000
            return LinearSVC(**kwargs)
        elif ctype == "ridge":
            return RidgeClassifier(**kwargs)
        else:
            self.raise_error(
                f"Unknown classifier_type '{self.classifier_type}'. "
                f"Supported types: 'logistic_regression', 'linear_svc', 'ridge'."
            )

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
        Plots the training data before classifier training:
        1. In the generative factor space showing sampled points per category.
        2. In the latent difference space (|z_1 - z_2|) colored by category.
        """
        factor_pairs = diagnostic_info if isinstance(diagnostic_info, dict) else {}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(factor_names))))

        # Left panel: Generative factor space
        f_x_name = factor_names[0]
        f_y_name = factor_names[1] if len(factor_names) > 1 else factor_names[0]
        f_x = factor_dict[f_x_name]
        f_y = factor_dict[f_y_name]

        # Background samples
        bg_subsample = min(5000, len(f_x))
        bg_idx = np.random.choice(len(f_x), size=bg_subsample, replace=False)
        ax1.scatter(f_x[bg_idx], f_y[bg_idx], c='lightgray', s=8, alpha=0.3, label='Dataset Samples')

        for k_idx, fname in enumerate(factor_names):
            if fname not in factor_pairs:
                continue
            orig_i, orig_j = factor_pairs[fname]
            if len(orig_i) == 0:
                continue

            num_display_pairs = min(500, len(orig_i))
            disp_idx = np.random.choice(len(orig_i), size=num_display_pairs, replace=False)
            pair_points = np.unique(np.concatenate([orig_i[disp_idx], orig_j[disp_idx]]))

            color = colors[k_idx % len(colors)]
            ax1.scatter(f_x[pair_points], f_y[pair_points], color=color, s=16, alpha=0.7, label=f"Fixed '{fname}' points")

        ax1.set_xlabel(f"Generative Factor: {f_x_name}", fontsize=10)
        ax1.set_ylabel(f"Generative Factor: {f_y_name}", fontsize=10)
        ax1.set_title("Sampled Points in Generative Factor Space", fontsize=11)
        ax1.legend(loc='best', fontsize=8)

        # Right panel: Latent difference space (|z_1 - z_2|)
        if latent_dim_names is not None and len(latent_dim_names) > 0:
            ld_x_label = latent_dim_names[0]
            ld_y_label = latent_dim_names[1] if len(latent_dim_names) > 1 else latent_dim_names[0]
        else:
            ld_x_label = "LD_1" if latent_array.shape[1] > 0 else "Dim 0"
            ld_y_label = "LD_2" if latent_array.shape[1] > 1 else ("LD_1" if latent_array.shape[1] > 0 else "Dim 0")

        z_x = X[:, 0]
        z_y = X[:, 1] if X.shape[1] > 1 else X[:, 0]

        for k_idx, fname in enumerate(factor_names):
            mask = y == k_idx
            if np.sum(mask) == 0:
                continue

            color = colors[k_idx % len(colors)]
            pts_x = z_x[mask]
            pts_y = z_y[mask]

            num_disp = min(1000, len(pts_x))
            disp_sub = np.random.choice(len(pts_x), size=num_disp, replace=False)

            ax2.scatter(
                pts_x[disp_sub],
                pts_y[disp_sub],
                color=color,
                s=12,
                alpha=0.5,
                label=f"Class: Fixed '{fname}'",
            )

        ax2.set_xlabel(f"Latent Difference |Δ{ld_x_label}|", fontsize=10)
        ax2.set_ylabel(f"Latent Difference |Δ{ld_y_label}|", fontsize=10)
        ax2.set_title("Classifier Input Features (|z_1 - z_2|)", fontsize=11)
        ax2.legend(loc='best', fontsize=8)

        plt.tight_layout()
        self.log_image(fig, "training_points", subpath="data")
        plt.close(fig)
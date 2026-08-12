import os
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from collective_encoder.testplotters.base import BaseTestPlotter
from collective_encoder.testplotters.transforms import add_transformed


class BetaMetric(BaseTestPlotter):
    """
    This implements a test metric for disentangled representations.
    The theoretical basis is given in Higgins et al. (ICLR 2017):
    https://papers.baulab.info/papers/also/Higgins-2017.pdf

    It evaluates how well latent dimensions capture independent generative factors
    by constructing difference vectors |z_1 - z_2| for sample pairs where a specific
    ground-truth factor is kept constant while other factors vary, and training a linear
    classifier to identify which generative factor was held constant.
    """
    _IDENTIFIER = "BetaMetric"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'generative_factors': None,
        'latent_dimensions': None,
        'factor_tolerances': 0.05,
        'variation_threshold': 0.1,
        'max_pairs_per_factor': 10000,
        'batch_size_L': 1,
        'min_samples_warning': 200,
        'test_split': 0.3,
        'num_models': 10,
        'confidence_interval': 0.95,
        'classifier_type': 'logistic_regression',
        'classifier_kwargs': None,
        'plot_confusion_matrix': True,
        'plot_training_points': False,
    })

    def __init__(self, args: Dict[str, Any] = None, **kwargs):
        super().__init__(args, **kwargs)
        
        self.data_dir = os.path.join(self.outpath, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.results_file = os.path.join(self.outpath, "results.txt")

    def collection_list(self) -> List[str]:
        return ["data", "labels", "latent", "meta"]

    def plot(self, data, latent, pred, labels, meta) -> None:
        """
        Calculates the Beta-VAE disentanglement metric and logs the results and confusion matrix.
        """
        labels = self._parse_selection(self.labels_selection, labels, "labels")
        latent = self._parse_selection(self.latents_selection, latent, "latent")
        meta = self._parse_selection(self.meta_selection, meta, "meta")

        vals = {}
        if isinstance(labels, dict):
            vals.update(labels)
        if isinstance(latent, dict):
            vals.update(latent)
        if isinstance(meta, dict):
            vals.update(meta)

        vals = add_transformed(self.transformed_values, vals)

        # Resolve generative factors
        if self.generative_factors is not None:
            if isinstance(self.generative_factors, str):
                factor_names = self.generative_factors.split(':')
            elif isinstance(self.generative_factors, (list, tuple)):
                factor_names = list(self.generative_factors)
            else:
                self.raise_error(f"Unsupported type for generative_factors: {type(self.generative_factors)}")
            for f in factor_names:
                if f not in vals:
                    self.raise_error(
                        f"Generative factor '{f}' specified in generative_factors not found. "
                        f"Available keys: {list(vals.keys())}"
                    )
            factor_dict = {f: np.squeeze(np.asarray(vals[f])) for f in factor_names}
        else:
            factor_dict = self._extract_factors_dict(labels)

        # Resolve latent dimensions
        if self.latent_dimensions is not None:
            if isinstance(self.latent_dimensions, str):
                ld_names = self.latent_dimensions.split(':')
            elif isinstance(self.latent_dimensions, (list, tuple)):
                ld_names = list(self.latent_dimensions)
            else:
                self.raise_error(f"Unsupported type for latent_dimensions: {type(self.latent_dimensions)}")
            for ld in ld_names:
                if ld not in vals:
                    self.raise_error(
                        f"Latent dimension '{ld}' specified in latent_dimensions not found. "
                        f"Available keys: {list(vals.keys())}"
                    )
            latent_arrays = [np.squeeze(np.asarray(vals[ld])) for ld in ld_names]
            if any(arr.ndim != 1 for arr in latent_arrays):
                self.raise_error("All selected latent dimensions must be 1-dimensional.")
            latent_array = np.column_stack(latent_arrays)
            latent_dim_names = ld_names
        else:
            latent_array = self._extract_latent_array(latent)
            if isinstance(latent, dict):
                latent_dim_names = list(latent.keys())
            else:
                latent_dim_names = [f"LD_{i+1}" for i in range(latent_array.shape[1])]

        if len(factor_dict) < 2:
            self.raise_error(
                f"BetaMetric requires at least 2 generative factors to form a "
                f"classification task, but found {len(factor_dict)} factors: {list(factor_dict.keys())}."
            )

        num_samples = latent_array.shape[0]
        for factor_name, factor_vals in factor_dict.items():
            if factor_vals.shape[0] != num_samples:
                self.raise_error(
                    f"Sample count mismatch: factor '{factor_name}' has length {factor_vals.shape[0]}, "
                    f"but latent array has {num_samples} samples."
                )

        factor_names = list(factor_dict.keys())
        self.log_info(
            f"Starting BetaMetric evaluation with {len(factor_names)} factors: {factor_names} "
            f"and {num_samples} latent representations ({latent_dim_names}) of dimension {latent_array.shape[1]}."
        )

        z_diff, y_diff, factor_sample_counts, factor_pairs = self._construct_difference_dataset(
            factor_dict=factor_dict,
            latent_array=latent_array,
        )

        if z_diff.shape[0] == 0 or len(np.unique(y_diff)) < 2:
            self.raise_error(
                f"Insufficient difference samples generated across categories. "
                f"Total samples: {z_diff.shape[0]}, unique classes: {np.unique(y_diff)}. "
                f"Consider adjusting 'factor_tolerances' or 'variation_threshold'."
            )

        if getattr(self, "plot_training_points", False):
            self._plot_and_save_training_points(
                factor_dict=factor_dict,
                latent_array=latent_array,
                factor_pairs=factor_pairs,
                z_diff=z_diff,
                y_diff=y_diff,
                factor_names=factor_names,
                latent_dim_names=latent_dim_names,
            )

        results = self._train_and_evaluate(
            z_diff=z_diff,
            y_diff=y_diff,
            factor_names=factor_names,
            factor_sample_counts=factor_sample_counts,
        )

        self._log_and_save_results(results, factor_names)

        if self.plot_confusion_matrix and results.get("confusion_matrix_mean") is not None:
            self._plot_and_save_confusion_matrix(
                cm_mean=results["confusion_matrix_mean"],
                factor_names=results["evaluated_factors"],
                accuracy_stats=results["test_accuracy_stats"],
                num_models=results["num_models"],
            )

    def _extract_factors_dict(self, labels: Any) -> Dict[str, np.ndarray]:
        """Ensures labels is converted to a dictionary of 1D numpy arrays."""
        if isinstance(labels, dict):
            factor_dict = {}
            for k, v in labels.items():
                arr = np.asarray(v)
                if arr.ndim > 1:
                    arr = np.squeeze(arr)
                if arr.ndim != 1:
                    self.raise_error(f"Factor '{k}' in labels must be 1-dimensional, got shape {arr.shape}.")
                factor_dict[str(k)] = arr
            return factor_dict
        elif isinstance(labels, np.ndarray):
            if labels.ndim == 1:
                return {"factor_0": labels}
            elif labels.ndim == 2:
                return {f"factor_{i}": labels[:, i] for i in range(labels.shape[1])}
            else:
                self.raise_error(f"Labels array must be 1D or 2D, got shape {labels.shape}.")
        else:
            self.raise_error(f"Unsupported type for labels: {type(labels)}. Expected dict or numpy array.")

    def _extract_latent_array(self, latent: Any) -> np.ndarray:
        """Ensures latent representations are formatted as a 2D numpy array (N, D)."""
        if isinstance(latent, dict):
            arrays = [np.asarray(v) for v in latent.values()]
            if all(a.ndim == 1 for a in arrays):
                return np.column_stack(arrays)
            elif all(a.ndim == 2 for a in arrays):
                return np.concatenate(arrays, axis=1)
            else:
                self.raise_error("Latent dictionary contains incompatible array shapes.")
        elif isinstance(latent, np.ndarray):
            if latent.ndim == 1:
                return latent[:, np.newaxis]
            elif latent.ndim == 2:
                return latent
            else:
                self.raise_error(f"Latent array must be 1D or 2D, got shape {latent.shape}.")
        else:
            self.raise_error(f"Unsupported type for latent: {type(latent)}. Expected dict or numpy array.")

    def _get_factor_tolerance(self, factor_name: str) -> float:
        if isinstance(self.factor_tolerances, dict):
            return float(self.factor_tolerances.get(factor_name, 0.05))
        return float(self.factor_tolerances)

    def _get_variation_threshold(self, factor_name: str) -> float:
        if isinstance(self.variation_threshold, dict):
            return float(self.variation_threshold.get(factor_name, 0.1))
        return float(self.variation_threshold)

    def _construct_difference_dataset(
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
                    # Enforce ordering to avoid duplicates
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

    def _plot_and_save_training_points(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
        factor_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        z_diff: np.ndarray,
        y_diff: np.ndarray,
        factor_names: List[str],
        latent_dim_names: Optional[List[str]] = None,
    ) -> None:
        """
        Plots the training data before classifier training:
        1. In the generative factor space showing sampled pairs per category.
        2. In the latent difference space (|z_1 - z_2|) colored by category.
        """
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

            # Subsample points for clean display
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

        z_x = z_diff[:, 0]
        z_y = z_diff[:, 1] if z_diff.shape[1] > 1 else z_diff[:, 0]

        for k_idx, fname in enumerate(factor_names):
            mask = y_diff == k_idx
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

    def _compute_ci(self, values: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculates mean, std, and confidence interval for a list of metric values."""
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "ci": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        mean_val = float(np.mean(arr))
        if n == 1:
            return {"mean": mean_val, "std": 0.0, "ci": 0.0, "ci_low": mean_val, "ci_high": mean_val}
        std_val = float(np.std(arr, ddof=1))
        std_err = std_val / np.sqrt(n)
        try:
            from scipy import stats
            t_crit = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df=n - 1))
        except Exception:
            t_crit = 1.96
        ci_val = float(t_crit * std_err)
        return {
            "mean": mean_val,
            "std": std_val,
            "ci": ci_val,
            "ci_low": mean_val - ci_val,
            "ci_high": mean_val + ci_val,
        }

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

    def _train_and_evaluate(
        self,
        z_diff: np.ndarray,
        y_diff: np.ndarray,
        factor_names: List[str],
        factor_sample_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Trains multiple linear classifiers with random seeds and calculates mean and CI."""
        unique_classes = np.unique(y_diff)
        evaluated_factors = [factor_names[c] for c in unique_classes]

        class_counts = {factor_names[c]: int(np.sum(y_diff == c)) for c in unique_classes}
        can_stratify = all(cnt >= 2 for cnt in class_counts.values())

        num_models = max(1, int(getattr(self, "num_models", 10)))
        conf_level = float(getattr(self, "confidence_interval", 0.95))

        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31 - 1, size=num_models)

        all_test_acc = []
        all_train_acc = []
        all_cm = []
        factor_test_accs = {fname: [] for fname in evaluated_factors}
        train_sizes = []
        test_sizes = []

        for r_idx, seed_val in enumerate(seeds):
            z_train, z_test, y_train, y_test = train_test_split(
                z_diff,
                y_diff,
                test_size=self.test_split,
                random_state=int(seed_val),
                stratify=y_diff if can_stratify else None,
            )
            train_sizes.append(z_train.shape[0])
            test_sizes.append(z_test.shape[0])

            clf = self._get_classifier(seed=seed_val)
            clf.fit(z_train, y_train)

            y_train_pred = clf.predict(z_train)
            y_test_pred = clf.predict(z_test)

            all_train_acc.append(float(accuracy_score(y_train, y_train_pred)))
            all_test_acc.append(float(accuracy_score(y_test, y_test_pred)))

            for c in unique_classes:
                fname = factor_names[c]
                mask = y_test == c
                if np.sum(mask) > 0:
                    factor_test_accs[fname].append(float(accuracy_score(y_test[mask], y_test_pred[mask])))

            cm = confusion_matrix(y_test, y_test_pred, labels=unique_classes)
            all_cm.append(cm)

        test_stats = self._compute_ci(all_test_acc, confidence_level=conf_level)
        train_stats = self._compute_ci(all_train_acc, confidence_level=conf_level)

        per_factor_stats = {}
        for fname in evaluated_factors:
            if len(factor_test_accs[fname]) > 0:
                per_factor_stats[fname] = self._compute_ci(factor_test_accs[fname], confidence_level=conf_level)
            else:
                per_factor_stats[fname] = None

        cm_mean = np.mean(all_cm, axis=0)
        cm_std = np.std(all_cm, axis=0, ddof=1) if num_models > 1 else np.zeros_like(cm_mean)

        return {
            "num_models": num_models,
            "confidence_level": conf_level,
            "test_accuracy_stats": test_stats,
            "train_accuracy_stats": train_stats,
            "all_test_accuracies": np.array(all_test_acc),
            "all_train_accuracies": np.array(all_train_acc),
            "per_factor_stats": per_factor_stats,
            "confusion_matrix_mean": cm_mean,
            "confusion_matrix_std": cm_std,
            "evaluated_factors": evaluated_factors,
            "factor_sample_counts": factor_sample_counts,
            "total_difference_samples": int(z_diff.shape[0]),
            "train_samples": int(train_sizes[0]),
            "test_samples": int(test_sizes[0]),
        }

    def _log_and_save_results(self, results: Dict[str, Any], factor_names: List[str]) -> None:
        """Logs the metric results to console, result file, and logger."""
        test_stats = results["test_accuracy_stats"]
        train_stats = results["train_accuracy_stats"]
        per_factor_stats = results["per_factor_stats"]
        sample_counts = results["factor_sample_counts"]
        num_models = results["num_models"]
        conf_pct = int(results["confidence_level"] * 100)

        t_mean, t_std, t_ci = test_stats["mean"], test_stats["std"], test_stats["ci"]
        tr_mean, tr_std, tr_ci = train_stats["mean"], train_stats["std"], train_stats["ci"]

        self.log_result_msg("=" * 60)
        self.log_result_msg(
            f"BetaMetric (Disentanglement Score): {t_mean:.4f} ± {t_std:.4f} "
            f"({conf_pct}% CI: [{t_mean - t_ci:.4f}, {t_mean + t_ci:.4f}])"
        )
        self.log_result_msg(
            f"Train Accuracy: {tr_mean:.4f} ± {tr_std:.4f} "
            f"({conf_pct}% CI: [{tr_mean - tr_ci:.4f}, {tr_mean + tr_ci:.4f}])"
        )
        self.log_result_msg(f"Number of Models Trained: {num_models}")
        self.log_result_msg(
            f"Total Difference Samples: {results['total_difference_samples']} "
            f"(Train: {results['train_samples']}, Test: {results['test_samples']})"
        )
        self.log_result_msg("-" * 60)
        self.log_result_msg(f"Sample Counts & Per-Factor Accuracy ({conf_pct}% CI):")
        for fname in factor_names:
            cnt = sample_counts.get(fname, 0)
            f_stat = per_factor_stats.get(fname)
            if f_stat is not None:
                fm, fstd, fci = f_stat["mean"], f_stat["std"], f_stat["ci"]
                acc_str = f"{fm * 100:.2f}% ± {fstd * 100:.2f}% (CI: [{(fm - fci) * 100:.2f}%, {(fm + fci) * 100:.2f}%])"
            else:
                acc_str = "N/A (no test samples)"
            warn_str = " [WARNING: < 200 samples]" if cnt < self.min_samples_warning else ""
            self.log_result_msg(f"  - Factor '{fname}': {cnt} samples | Accuracy: {acc_str}{warn_str}")
        self.log_result_msg("=" * 60)

        score_fn = os.path.join(self.data_dir, "beta_disentanglement_score.npy")
        np.save(score_fn, results["all_test_accuracies"])
        self.log_info(f"Saved data 'beta_disentanglement_score' to {score_fn}")

        if results.get("confusion_matrix_mean") is not None:
            cm_fn = os.path.join(self.data_dir, "confusion_matrix.npy")
            np.save(cm_fn, results["confusion_matrix_mean"])
            self.log_info(f"Saved data 'confusion_matrix' to {cm_fn}")

        if self.logger_type == "WandbLogger" and self.logger is not None:
            try:
                self.logger.experiment.log({
                    f"[{type(self).__name__}] disentanglement_score_mean": t_mean,
                    f"[{type(self).__name__}] disentanglement_score_std": t_std,
                    f"[{type(self).__name__}] train_accuracy_mean": tr_mean,
                })
            except Exception as e:
                self.log_warn(f"Failed to log metric score to WandbLogger: {e}")

    def _plot_and_save_confusion_matrix(
        self,
        cm_mean: np.ndarray,
        factor_names: List[str],
        accuracy_stats: Dict[str, float],
        num_models: int,
    ) -> None:
        """Plots and saves the averaged confusion matrix across all trained models."""
        n_classes = len(factor_names)
        fig_size = max(5, n_classes * 1.2)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        im = ax.imshow(cm_mean, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, label="Average Sample Count")

        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(factor_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(factor_names, fontsize=9)

        ax.set_xlabel("Predicted Factor", fontsize=10, labelpad=8)
        ax.set_ylabel("True Fixed Factor", fontsize=10, labelpad=8)

        acc_mean = accuracy_stats["mean"]
        acc_ci = accuracy_stats["ci"]
        ax.set_title(
            f"BetaMetric Confusion Matrix\n"
            f"(Accuracy: {acc_mean * 100:.2f}% ± {acc_ci * 100:.2f}% 95% CI, {num_models} models)",
            fontsize=10,
            pad=12,
        )

        thresh = cm_mean.max() / 2.0 if cm_mean.max() > 0 else 1.0
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_mean[i, j]
                text_str = f"{val:.0f}" if abs(val - round(val)) < 1e-4 else f"{val:.1f}"
                ax.text(
                    j, i, text_str,
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=9,
                )

        plt.tight_layout()
        self.log_image(fig, "confusion_matrix")
        plt.close(fig)
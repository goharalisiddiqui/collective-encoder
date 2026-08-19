import os
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from collective_encoder.testplotters.base import BaseTestPlotter
from collective_encoder.testplotters.transforms import add_transformed


class BaseDisentanglementMetric(BaseTestPlotter):
    """
    Abstract base class for unsupervised / supervised disentanglement metrics.
    Provides a unified pipeline for:
      - Variable collection and selection parsing (labels, latents, metadata, transforms)
      - Representation formatting and validation
      - Shared information-theoretic computations (discrete entropy, mutual information)
      - Multi-model evaluation with independent random seeds
      - Student-t confidence interval calculation
      - Standardized reporting to results.txt and .npy export
      - Heatmap / Confusion matrix rendering and output directory organization
    """
    _IDENTIFIER = "BaseDisentanglementMetric"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'generative_factors': None,
        'latent_dimensions': None,
        'factor_tolerances': 0.05,
        'variation_threshold': 0.1,
        'min_samples_warning': 200,
        'test_split': 0.3,
        'num_models': 10,
        'confidence_interval': 0.95,
        'plot_confusion_matrix': True,
        'plot_training_points': False,
    })

    def __init__(self, args: Dict[str, Any] = None, **kwargs):
        super().__init__(args, **kwargs)
        
        # Dedicated data folder for .npy datasets and data plots
        self.data_dir = os.path.join(self.outpath, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.results_file = os.path.join(self.outpath, "results.txt")

    def collection_list(self) -> List[str]:
        return ["data", "labels", "latent", "meta"]

    def plot(self, data, latent, pred, labels, meta) -> None:
        """
        Main execution pipeline for disentanglement evaluation.
        Parses inputs and dispatches to _run_evaluation.
        """
        try:
            labels = self._parse_selection(self.labels_selection, labels, "labels")
            latent = self._parse_selection(self.latents_selection, latent, "latent")
            meta = self._parse_selection(self.meta_selection, meta, "meta")
        except Exception as e:
            self.log_exception(f"Error occurred while parsing selections: {e}")
            return

        vals = {}
        if isinstance(labels, dict):
            vals.update(labels)
        if isinstance(latent, dict):
            vals.update(latent)
        if isinstance(meta, dict):
            vals.update(meta)

        vals = add_transformed(self.transformed_values, vals)

        # Resolve ground-truth generative factors
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

        # Resolve latent dimensions (features used for evaluation)
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
                f"{type(self).__name__} requires at least 2 generative factors to evaluate disentanglement, "
                f"but found {len(factor_dict)} factors: {list(factor_dict.keys())}."
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
            f"Starting {type(self).__name__} evaluation with {len(factor_names)} factors: {factor_names} "
            f"and {num_samples} latent representations ({latent_dim_names}) of dimension {latent_array.shape[1]}."
        )

        self._run_evaluation(
            factor_dict=factor_dict,
            latent_array=latent_array,
            factor_names=factor_names,
            latent_dim_names=latent_dim_names,
        )

    def _run_evaluation(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
        factor_names: List[str],
        latent_dim_names: List[str],
    ) -> None:
        """
        Default execution pipeline for classification-based metrics (e.g. Beta, Factor).
        Metrics based on information theory or regression can override this method.
        """
        X, y, factor_sample_counts, diagnostic_info = self._construct_dataset(
            factor_dict=factor_dict,
            latent_array=latent_array,
        )

        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            self.raise_error(
                f"Insufficient samples generated across categories. "
                f"Total samples: {X.shape[0]}, unique classes: {np.unique(y)}. "
                f"Consider adjusting 'factor_tolerances' or 'variation_threshold'."
            )

        if getattr(self, "plot_training_points", False):
            self._plot_and_save_training_points(
                factor_dict=factor_dict,
                latent_array=latent_array,
                diagnostic_info=diagnostic_info,
                X=X,
                y=y,
                factor_names=factor_names,
                latent_dim_names=latent_dim_names,
            )

        results = self._train_and_evaluate(
            X=X,
            y=y,
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

    # -------------------------------------------------------------------------
    # Abstract / Subclass Interface Methods
    # -------------------------------------------------------------------------
    def _construct_dataset(
        self,
        factor_dict: Dict[str, np.ndarray],
        latent_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Any]:
        """Subclasses for pair/batch classification implement this method."""
        raise NotImplementedError("Subclasses must implement _construct_dataset.")

    def _get_classifier(self, seed: Optional[int] = None) -> Any:
        """Subclasses for classification implement this method."""
        raise NotImplementedError("Subclasses must implement _get_classifier.")

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
        """Subclasses implement metric-specific diagnostic visualizations."""
        pass

    # -------------------------------------------------------------------------
    # Shared Helper Methods: Data Extraction & Validation
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Shared Helper Methods: Information Theory & Statistics
    # -------------------------------------------------------------------------
    def _discretize_variable(self, arr: np.ndarray, num_bins: int = 20) -> np.ndarray:
        """Discretizes a 1D continuous array into integer bin indices [0, num_bins - 1]."""
        arr_1d = np.squeeze(np.asarray(arr, dtype=float))
        if arr_1d.ndim != 1:
            arr_1d = arr_1d.flatten()
        # If array has very few unique values, map them directly to integer categories
        unique_vals = np.unique(arr_1d)
        if len(unique_vals) <= num_bins:
            _, inverse = np.unique(arr_1d, return_inverse=True)
            return inverse

        min_val, max_val = float(np.min(arr_1d)), float(np.max(arr_1d))
        if abs(max_val - min_val) < 1e-12:
            return np.zeros(len(arr_1d), dtype=int)

        bins = np.linspace(min_val, max_val, num_bins + 1)
        bin_indices = np.digitize(arr_1d, bins[:-1]) - 1
        return np.clip(bin_indices, 0, num_bins - 1).astype(int)

    def _compute_discrete_entropy(self, y: np.ndarray, num_bins: int = 20) -> float:
        """Computes empirical Shannon entropy H(Y) in nats."""
        y_discrete = self._discretize_variable(y, num_bins=num_bins)
        _, counts = np.unique(y_discrete, return_counts=True)
        probs = counts.astype(float) / np.sum(counts)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def _compute_discrete_mutual_info(self, x: np.ndarray, y: np.ndarray, num_bins: int = 20) -> float:
        """Computes empirical mutual information I(X; Y) between two 1D variables using 2D histogram binning."""
        x_d = self._discretize_variable(x, num_bins=num_bins)
        y_d = self._discretize_variable(y, num_bins=num_bins)

        # Build 2D joint histogram
        hist_2d, _, _ = np.histogram2d(x_d, y_d, bins=[num_bins, num_bins], range=[[0, num_bins], [0, num_bins]])
        p_xy = hist_2d / np.sum(hist_2d)

        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # Non-zero joint probabilities
        mask = p_xy > 0
        p_xy_nz = p_xy[mask]
        p_x_nz = p_x[np.where(mask)[0]]
        p_y_nz = p_y[np.where(mask)[1]]

        mi = np.sum(p_xy_nz * np.log(p_xy_nz / (p_x_nz * p_y_nz)))
        return float(max(0.0, mi))

    def _compute_mutual_info_matrix(
        self,
        latents: np.ndarray,
        factors_dict: Dict[str, np.ndarray],
        num_bins: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Computes mutual information matrix I \in R^{D x K} and factor entropy vector H \in R^K.
        """
        factor_names = list(factors_dict.keys())
        num_latents = latents.shape[1]
        num_factors = len(factor_names)

        mi_matrix = np.zeros((num_latents, num_factors), dtype=float)
        factor_entropies = np.zeros(num_factors, dtype=float)

        for k_idx, fname in enumerate(factor_names):
            f_vals = factors_dict[fname]
            factor_entropies[k_idx] = self._compute_discrete_entropy(f_vals, num_bins=num_bins)
            for d_idx in range(num_latents):
                z_vals = latents[:, d_idx]
                mi_matrix[d_idx, k_idx] = self._compute_discrete_mutual_info(z_vals, f_vals, num_bins=num_bins)

        return mi_matrix, factor_entropies

    def _compute_ci(self, values: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculates mean, std, and Student-t confidence interval for a list of metric values."""
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

    # -------------------------------------------------------------------------
    # Shared Helper Methods: Multi-Model Evaluation & Classification
    # -------------------------------------------------------------------------
    def _train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        factor_names: List[str],
        factor_sample_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Trains multiple classifiers with random seeds and calculates mean and CI."""
        unique_classes = np.unique(y)
        evaluated_factors = [factor_names[c] for c in unique_classes]

        class_counts = {factor_names[c]: int(np.sum(y == c)) for c in unique_classes}
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
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_split,
                random_state=int(seed_val),
                stratify=y if can_stratify else None,
            )
            train_sizes.append(X_train.shape[0])
            test_sizes.append(X_test.shape[0])

            clf = self._get_classifier(seed=seed_val)
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

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
            "total_samples": int(X.shape[0]),
            "train_samples": int(train_sizes[0]),
            "test_samples": int(test_sizes[0]),
        }

    # -------------------------------------------------------------------------
    # Shared Helper Methods: Logging & Visualization
    # -------------------------------------------------------------------------
    def _log_and_save_results(self, results: Dict[str, Any], factor_names: List[str]) -> None:
        """Logs classification metric results to console, result file, and logger."""
        metric_name = type(self).__name__
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
            f"{metric_name} (Disentanglement Score): {t_mean:.4f} ± {t_std:.4f} "
            f"({conf_pct}% CI: [{t_mean - t_ci:.4f}, {t_mean + t_ci:.4f}])"
        )
        self.log_result_msg(
            f"Train Accuracy: {tr_mean:.4f} ± {tr_std:.4f} "
            f"({conf_pct}% CI: [{tr_mean - tr_ci:.4f}, {tr_mean + tr_ci:.4f}])"
        )
        self.log_result_msg(f"Number of Models Trained: {num_models}")
        self.log_result_msg(
            f"Total Evaluation Samples: {results['total_samples']} "
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

        score_fn = os.path.join(self.data_dir, "disentanglement_score.npy")
        np.save(score_fn, results["all_test_accuracies"])
        self.log_info(f"Saved data 'disentanglement_score' to {score_fn}")

        if results.get("confusion_matrix_mean") is not None:
            cm_fn = os.path.join(self.data_dir, "confusion_matrix.npy")
            np.save(cm_fn, results["confusion_matrix_mean"])
        # Register metrics into metrics dictionary
        self.set_metric("disentanglement_score", t_mean)
        self.set_metric("test_accuracy", t_mean)
        self.set_metric("train_accuracy", tr_mean)
        self.set_metric("disentanglement_score_std", t_std)

        if self.logger_type == "WandbLogger" and self.logger is not None:
            try:
                self.logger.experiment.log({
                    f"[{metric_name}] disentanglement_score_mean": t_mean,
                    f"[{metric_name}] disentanglement_score_std": t_std,
                    f"[{metric_name}] train_accuracy_mean": tr_mean,
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
        metric_name = type(self).__name__
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
            f"{metric_name} Confusion Matrix\n"
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

    def _plot_and_save_matrix_heatmap(
        self,
        matrix: np.ndarray,
        row_names: List[str],
        col_names: List[str],
        xlabel: str,
        ylabel: str,
        title: str,
        image_name: str,
        cbar_label: str = "",
        cmap: str = "Blues",
        subpath: Optional[str] = None,
        val_format: str = "{:.3f}",
    ) -> None:
        """Reusable heatmap visualizer for importance, mutual info, or score matrices."""
        n_rows, n_cols = matrix.shape
        fig_w = max(5.5, n_cols * 1.3)
        fig_h = max(4.5, n_rows * 0.9)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(matrix, interpolation='nearest', cmap=cmap, aspect='auto')
        fig.colorbar(im, ax=ax, label=cbar_label)

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(col_names, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(row_names, fontsize=9)

        ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
        ax.set_title(title, fontsize=11, pad=12)

        thresh = (matrix.max() + matrix.min()) / 2.0 if matrix.max() > matrix.min() else matrix.max() / 2.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                ax.text(
                    j, i, val_format.format(val),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=9,
                )

        plt.tight_layout()
        self.log_image(fig, image_name, subpath=subpath)
        plt.close(fig)

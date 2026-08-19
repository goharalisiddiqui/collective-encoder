"""Unified Disentanglement TestPlotter running all disentanglement metrics."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from collective_encoder.testplotters.base import BaseTestPlotter
from collective_encoder.testplotters.disentanglement_metrics.beta import DisentanglementBetaMetric
from collective_encoder.testplotters.disentanglement_metrics.factor import DisentanglementFactorMetric
from collective_encoder.testplotters.disentanglement_metrics.dci import DisentanglementDCIMetric
from collective_encoder.testplotters.disentanglement_metrics.mig import DisentanglementMIGMetric
from collective_encoder.testplotters.disentanglement_metrics.sap import DisentanglementSAPMetric
from collective_encoder.testplotters.disentanglement_metrics.modularity import DisentanglementModularityMetric

_log = logging.getLogger(__name__)

_METRIC_MAP: Dict[str, Any] = {
    "beta": DisentanglementBetaMetric,
    "factor": DisentanglementFactorMetric,
    "dci": DisentanglementDCIMetric,
    "mig": DisentanglementMIGMetric,
    "sap": DisentanglementSAPMetric,
    "modularity": DisentanglementModularityMetric,
}

_ALL_METRIC_KEYS = ["beta", "factor", "dci", "mig", "sap", "modularity"]


class DisentanglementPlotter(BaseTestPlotter):
    r"""
    Unified test plotter that runs all standard disentanglement metrics in a single pass:
      1. Beta-VAE Metric (Higgins et al., ICLR 2017)
      2. Factor-VAE Metric (Kim & Mnih, ICML 2018)
      3. DCI Disentanglement, Completeness, Informativeness (Eastwood & Williams, ICLR 2018)
      4. Mutual Information Gap (MIG) (Chen et al., NeurIPS 2018)
      5. Separated Attribute Predictability (SAP) (Kumar et al., ICLR 2018)
      6. Modularity & Explicitness (Ridgeway & Mozer, NeurIPS 2018)

    Features:
      - Automatically selects all latent space dimensions as `latent_dimensions` (LD_1..LD_D).
      - Automatically selects all ground-truth labels as `generative_factors`.
      - Supports `name` parameter to register metrics as `<name>_<metric>` (e.g. `DIS_beta`, `DIS_mig`).
      - Shared keywords (e.g. `confidence_interval`, `num_models`, `test_split`, `num_bins`)
        can be defined once at the top level.
      - Individual metrics can override any shared parameter via `<metric_name>_<keyword>`
        (e.g. `beta_confidence_interval: 0.99`).
      - Metric-specific keywords are passed using their prefixed names
        (e.g. `beta_max_pairs_per_factor`, `dci_regressor_type`).

    Example YAML configuration:
    ```yaml
    test_plotters:
      - tester_type: DisentanglementPlotter
        tester_args:
          name: "DIS"
          metrics: ["beta", "factor", "dci", "mig", "sap", "modularity"]
          num_models: 10
          confidence_interval: 0.95
          test_split: 0.3
          beta_confidence_interval: 0.99
          dci_regressor_type: "gradient_boosting"
    ```
    """

    _IDENTIFIER = "DisentanglementPlotter"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'name': None,
        'metrics': ["beta", "factor", "dci", "mig", "sap", "modularity"],
        'generative_factors': None,
        'latent_dimensions': None,
        'confidence_interval': 0.95,
        'confidence_level': 0.95,
        'num_models': 10,
        'test_split': 0.3,
        'factor_tolerances': 0.05,
        'variation_threshold': 0.1,
        'min_samples_warning': 200,
        'num_bins': 20,
        'subsample_ratio': 0.8,
        'plot_confusion_matrix': True,
        'plot_training_points': False,
        'plot_importance_matrix': True,
        'plot_mi_matrix': True,
        'plot_sap_matrix': True,
    })

    def collection_list(self) -> List[str]:
        return ["data", "labels", "latent", "meta"]

    def plot(self, data, latent, pred, labels, meta) -> None:
        # 1. Automatically resolve generative factors if not specified
        factors = self._resolve_factors(labels=labels, meta=meta)

        # 2. Automatically resolve latent dimensions if not specified
        l_dims = self._resolve_latent_dims(latent=latent, meta=meta)

        # 3. Determine prefix name and which metrics to execute
        name = getattr(self, "name", None)
        if isinstance(name, str):
            name = name.strip()
            if not name:
                name = None

        raw_metrics = getattr(self, "metrics", _ALL_METRIC_KEYS)
        if isinstance(raw_metrics, str):
            selected_metrics = [m.strip().lower() for m in raw_metrics.split(":") if m.strip()]
        elif isinstance(raw_metrics, (list, tuple)):
            selected_metrics = [str(m).strip().lower() for m in raw_metrics if str(m).strip()]
        else:
            selected_metrics = _ALL_METRIC_KEYS

        self.log_result_msg("=" * 65)
        self.log_result_msg(" Starting Unified Disentanglement Evaluation Suite")
        if name:
            self.log_result_msg(f" Name Prefix         : {name}")
        self.log_result_msg(f" Metrics to Evaluate : {selected_metrics}")
        self.log_result_msg(f" Generative Factors  : {factors}")
        self.log_result_msg(f" Latent Dimensions   : {l_dims}")
        self.log_result_msg("=" * 65)

        summary_results: Dict[str, Dict[str, Any]] = {}

        # 4. Execute each metric
        for m_key in selected_metrics:
            if m_key not in _METRIC_MAP:
                self.log_warn(f"Unknown disentanglement metric '{m_key}'. Valid choices: {list(_METRIC_MAP.keys())}")
                continue

            metric_cls = _METRIC_MAP[m_key]
            sub_args = self._build_sub_metric_args(m_key, metric_cls, factors, l_dims)

            try:
                sub_plotter = metric_cls(args=sub_args, run_dir=self.run_dir)
                sub_plotter.outpath = os.path.join(self.outpath, m_key)
                os.makedirs(sub_plotter.outpath, exist_ok=True)
                if hasattr(sub_plotter, "data_dir"):
                    sub_plotter.data_dir = os.path.join(sub_plotter.outpath, "data")
                    os.makedirs(sub_plotter.data_dir, exist_ok=True)
                sub_plotter.results_file = os.path.join(sub_plotter.outpath, "results.txt")

                sub_plotter.plot(data=data, latent=latent, pred=pred, labels=labels, meta=meta)

                # Merge metrics into self.metrics_dict with optional prefix
                metric_outputs = sub_plotter.get_metrics()
                for k, v in metric_outputs.items():
                    self.set_metric(k, v)
                    if name:
                        self.set_metric(f"{name}_{k}", v)

                # Register primary metric short aliases (e.g. DIS_beta, DIS_mig, DIS_dci)
                primary_score = None
                if m_key == "beta":
                    primary_score = metric_outputs.get("beta_score")
                elif m_key == "factor":
                    primary_score = metric_outputs.get("factor_score")
                elif m_key == "dci":
                    primary_score = metric_outputs.get("dci_disentanglement")
                elif m_key == "mig":
                    primary_score = metric_outputs.get("mig_score")
                elif m_key == "sap":
                    primary_score = metric_outputs.get("sap_score")
                elif m_key == "modularity":
                    primary_score = metric_outputs.get("modularity_score")

                if primary_score is not None:
                    self.set_metric(m_key, primary_score)
                    if name:
                        self.set_metric(f"{name}_{m_key}", primary_score)

                summary_results[m_key] = metric_outputs

            except Exception as e:
                self.log_exception(f"Metric '{m_key}' failed with error: {e}")

        # 5. Write unified summary report
        self._write_unified_summary(summary_results, factors, l_dims, name=name)

    def _resolve_factors(self, labels: Any, meta: Any) -> List[str]:
        """Auto-discovers all available ground-truth factors if not explicitly specified."""
        if getattr(self, "generative_factors", None) is not None:
            gf = self.generative_factors
            if isinstance(gf, str):
                return [f.strip() for f in gf.split(":") if f.strip()]
            elif isinstance(gf, (list, tuple)):
                return [str(f).strip() for f in gf if str(f).strip()]

        if isinstance(labels, dict) and len(labels) > 0:
            return list(labels.keys())
        elif isinstance(labels, np.ndarray):
            n_factors = labels.shape[1] if labels.ndim > 1 else 1
            return [f"factor_{i}" for i in range(n_factors)]
        elif isinstance(meta, dict) and len(meta) > 0:
            return list(meta.keys())

        return []

    def _resolve_latent_dims(self, latent: Any, meta: Any) -> List[str]:
        """Auto-discovers all available latent dimensions if not explicitly specified."""
        if getattr(self, "latent_dimensions", None) is not None:
            ld = self.latent_dimensions
            if isinstance(ld, str):
                return [d.strip() for d in ld.split(":") if d.strip()]
            elif isinstance(ld, (list, tuple)):
                return [str(d).strip() for d in ld if str(d).strip()]

        if isinstance(latent, dict) and len(latent) > 0:
            return list(latent.keys())
        elif isinstance(latent, np.ndarray):
            n_dims = latent.shape[1] if latent.ndim > 1 else 1
            return [f"LD_{i+1}" for i in range(n_dims)]
        elif isinstance(meta, dict) and "mu_latent" in meta:
            arr = np.asarray(meta["mu_latent"])
            n_dims = arr.shape[1] if arr.ndim > 1 else 1
            return [f"LD_{i+1}" for i in range(n_dims)]

        return []

    def _build_sub_metric_args(
        self,
        m_name: str,
        metric_cls: Any,
        factors: List[str],
        l_dims: List[str],
    ) -> Dict[str, Any]:
        """
        Builds the argument dictionary for an individual sub-metric.
        Applies hierarchy:
          1. Metric-prefixed argument: `<metric_name>_<param>` (e.g. `beta_confidence_interval`)
          2. Top-level shared argument: `<param>` (e.g. `confidence_interval`)
          3. Metric class default value.
        """
        sub_args: Dict[str, Any] = {}
        raw_args = getattr(self, "args", {}) or {}

        # Default factors and latent dimensions
        sub_args["generative_factors"] = factors if factors else None
        sub_args["latent_dimensions"] = l_dims if l_dims else None

        # Pass selection and transform configurations
        for key in ["labels_selection", "latents_selection", "meta_selection", "transformed_values"]:
            if hasattr(self, key):
                sub_args[key] = getattr(self, key)

        # Inspect all optional args defined on the target metric class
        class_optional_args = getattr(metric_cls, "_OPTIONAL_ARGS", {})

        for param_key, default_val in class_optional_args.items():
            prefixed_key = f"{m_name}_{param_key}"

            if prefixed_key in raw_args:
                sub_args[param_key] = raw_args[prefixed_key]
            elif hasattr(self, prefixed_key):
                sub_args[param_key] = getattr(self, prefixed_key)
            elif param_key in raw_args:
                sub_args[param_key] = raw_args[param_key]
            elif hasattr(self, param_key):
                sub_args[param_key] = getattr(self, param_key)
            else:
                sub_args[param_key] = default_val

        # Support confidence_level as alias for confidence_interval
        if f"{m_name}_confidence_level" in raw_args:
            sub_args["confidence_interval"] = raw_args[f"{m_name}_confidence_level"]
        elif "confidence_level" in raw_args:
            sub_args["confidence_interval"] = raw_args["confidence_level"]

        return sub_args

    def _write_unified_summary(
        self,
        summary_results: Dict[str, Dict[str, Any]],
        factors: List[str],
        l_dims: List[str],
        name: Optional[str] = None,
    ) -> None:
        """Writes a formatted summary of all evaluated disentanglement metrics."""
        summary_file = os.path.join(self.outpath, "disentanglement_summary.txt")

        primary_scores = {
            "Beta-VAE Score": summary_results.get("beta", {}).get("beta_score"),
            "Factor-VAE Score": summary_results.get("factor", {}).get("factor_score"),
            "DCI Disentanglement": summary_results.get("dci", {}).get("dci_disentanglement"),
            "DCI Completeness": summary_results.get("dci", {}).get("dci_completeness"),
            "DCI Informativeness": summary_results.get("dci", {}).get("dci_informativeness"),
            "MIG Score": summary_results.get("mig", {}).get("mig_score"),
            "SAP Score": summary_results.get("sap", {}).get("sap_score"),
            "Modularity Score": summary_results.get("modularity", {}).get("modularity_score"),
            "Explicitness Score": summary_results.get("modularity", {}).get("explicitness_score"),
        }

        # Calculate average disentanglement score across available normalized metrics
        eval_scores = [
            v for k, v in primary_scores.items()
            if v is not None and k in ["Beta-VAE Score", "Factor-VAE Score", "DCI Disentanglement", "MIG Score", "SAP Score", "Modularity Score"]
        ]
        if eval_scores:
            avg_score = float(np.mean(eval_scores))
            self.set_metric("disentanglement_score_mean", avg_score)
            self.set_metric("disentanglement_mean", avg_score)
            if name:
                self.set_metric(f"{name}_mean", avg_score)
                self.set_metric(f"{name}_avg", avg_score)
                self.set_metric(f"{name}_score", avg_score)
                self.set_metric(f"{name}_disentanglement_score_mean", avg_score)
        else:
            avg_score = None

        lines = [
            "=" * 70,
            " UNIFIED DISENTANGLEMENT EVALUATION REPORT",
            "=" * 70,
            f" Name Prefix         : {name if name else 'None'}",
            f" Evaluated Factors ({len(factors)}): {factors}",
            f" Latent Dimensions ({len(l_dims)}): {l_dims}",
            "-" * 70,
            " Summary Scores:",
        ]

        for s_name, val in primary_scores.items():
            if val is not None:
                lines.append(f"   - {s_name:<26}: {val:.4f}")

        if avg_score is not None:
            lines.append(f"   - {'Mean Disentanglement Score':<26}: {avg_score:.4f}")

        lines.append("=" * 70)

        report_text = "\n".join(lines)
        self.log_result_msg("\n" + report_text)

        try:
            with open(summary_file, "w") as f:
                f.write(report_text + "\n")
        except Exception as e:
            self.log_warn(f"Could not save summary file {summary_file}: {e}")



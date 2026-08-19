import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import yaml
from gslibs.drivers.driverutils.slurm import (
    get_job_state,
    parse_sbatch_header,
    slurm_driver,
)

import collective_encoder.ce_run_base as crb
from collective_encoder.hyperparameter_optimization.resolver import ConfigResolver
from collective_encoder.hyperparameter_optimization.search_space import SearchSpace
from collective_encoder.trainer import run_training_experiment

_log = logging.getLogger(__name__)

# Backward-compatibility aliases
parse_slurm_header_file = parse_sbatch_header
get_slurm_job_state = get_job_state


def _make_paths_absolute(config: Dict[str, Any]) -> None:
    path_keys = ["load_datamodule", "load_model", "load_network", "dataset_path"]
    for k in path_keys:
        if k in config and isinstance(config[k], str) and config[k]:
            if not os.path.isabs(config[k]):
                config[k] = os.path.abspath(config[k])


def extract_metric(metric_name: str, results: Dict[str, Any], test_results: Dict[str, Any]) -> float:
    """Extracts a metric value from results/test_results.

    Supports:
      - Direct scalar metric names (e.g. 'val_loss', 'test_mae', 'beta_score')
      - Direct correlation names (e.g. 'C_LL', 'C_LL_abs', 'C_cross_mean', 'C_cross_mean_abs')
      - Named pair elements (e.g. 'C_cross_LD_1_phi', 'C_cross_LD_1_phi_abs')
      - Bracket label indexing (e.g. 'C_cross[LD_1, phi]', 'C_cross[LD_1][phi]')
      - Bracket numerical indexing (e.g. 'C_cross[0, 1]', 'C_cross[0][1]')
      - Absolute value wrappers (e.g. 'abs(C_cross[0, 1])')

    Raises:
      KeyError: If metric, matrix, or variable label is not found.
      IndexError: If row/column index is out of bounds for the correlation matrix.
    """
    is_abs = False
    name = metric_name.strip()
    if name.startswith("abs(") and name.endswith(")"):
        is_abs = True
        name = name[4:-1].strip()
    elif name.endswith("_abs") and not (isinstance(test_results, dict) and name in test_results):
        is_abs = True
        name = name[:-4].strip()

    # 1. Direct scalar match in results or test_results
    if name == "val_loss":
        val = results.get("val_loss", None)
        if val is not None and isinstance(val, (int, float)):
            return float(abs(val) if is_abs else val)
    if name in results and isinstance(results[name], (int, float)):
        return float(abs(results[name]) if is_abs else results[name])
    if isinstance(test_results, dict) and name in test_results and isinstance(test_results[name], (int, float)):
        return float(abs(test_results[name]) if is_abs else test_results[name])

    # 2. Check for bracket notation: e.g. "C_cross[0, 1]", "C_cross[LD_1, phi]", "C_cross[0][1]"
    bracket_match = re.match(r"^([a-zA-Z0-9_-]+)\[([^\]]+)\](?:\[([^\]]+)\])?$", name)
    if bracket_match:
        base_name = bracket_match.group(1)
        part1 = bracket_match.group(2).strip()
        part2 = bracket_match.group(3).strip() if bracket_match.group(3) else None

        if part2 is None and "," in part1:
            parts = [p.strip() for p in part1.split(",")]
            if len(parts) == 2:
                part1, part2 = parts[0], parts[1]

        # Look for matrix metadata in test_results or results
        matrix_key = f"{base_name}_matrix"
        matrix_data = None
        if isinstance(test_results, dict) and matrix_key in test_results:
            matrix_data = test_results[matrix_key]
        elif matrix_key in results:
            matrix_data = results[matrix_key]

        if matrix_data is not None and isinstance(matrix_data, dict):
            x_labels = matrix_data.get("x_labels", [])
            y_labels = matrix_data.get("y_labels", [])
            mat = np.array(matrix_data.get("matrix", []))

            # Resolve row index
            if part1.isdigit():
                row_idx = int(part1)
                if row_idx < 0 or row_idx >= mat.shape[0]:
                    raise IndexError(
                        f"Row index {row_idx} is out of bounds for matrix '{base_name}' of shape {mat.shape}. "
                        f"Valid row indices are 0..{mat.shape[0] - 1}."
                    )
            elif part1 in x_labels:
                row_idx = x_labels.index(part1)
            else:
                raise KeyError(
                    f"Row variable '{part1}' not found in matrix '{base_name}'. "
                    f"Available row variables: {x_labels} (or numerical indices 0..{len(x_labels)-1})."
                )

            # Resolve col index
            if part2 is not None:
                if part2.isdigit():
                    col_idx = int(part2)
                    if col_idx < 0 or col_idx >= mat.shape[1]:
                        raise IndexError(
                            f"Column index {col_idx} is out of bounds for matrix '{base_name}' of shape {mat.shape}. "
                            f"Valid column indices are 0..{mat.shape[1] - 1}."
                        )
                elif part2 in y_labels:
                    col_idx = y_labels.index(part2)
                else:
                    raise KeyError(
                        f"Column variable '{part2}' not found in matrix '{base_name}'. "
                        f"Available column variables: {y_labels} (or numerical indices 0..{len(y_labels)-1})."
                    )
            else:
                col_idx = 0

            val = float(mat[row_idx, col_idx])
            return float(abs(val) if is_abs else val)

        # Fallback to flattened naming conventions
        alt_key = f"{base_name}_{part1}_{part2}" if part2 else f"{base_name}_{part1}"
        if isinstance(test_results, dict) and alt_key in test_results:
            val = float(test_results[alt_key])
            return float(abs(val) if is_abs else val)

    # 3. Fuzzy/normalized matching (case-insensitive, ignoring underscores)
    if isinstance(test_results, dict):
        norm_target = name.lower().replace("_", "").replace("-", "")
        for k, v in test_results.items():
            if isinstance(v, (int, float)):
                norm_k = k.lower().replace("_", "").replace("-", "")
                if norm_k == norm_target:
                    return float(abs(v) if is_abs else v)

    # 4. Not found -> raise informative error
    avail = list(results.keys()) + (list(test_results.keys()) if isinstance(test_results, dict) else [])
    avail_clean = [k for k in avail if not k.endswith("_matrix")]
    raise KeyError(
        f"Requested metric '{metric_name}' not found in trial results or test results.\n"
        f"Available metrics: {avail_clean}\n"
        f"For correlation matrices, you can select elements using:\n"
        f"  - Named syntax: '<corr_name>_<x_label>_<y_label>' (e.g. 'C_cross_LD_1_phi')\n"
        f"  - Bracket label syntax: '<corr_name>[<x_label>, <y_label>]' (e.g. 'C_cross[LD_1, phi]')\n"
        f"  - Bracket index syntax: '<corr_name>[<row>, <col>]' (e.g. 'C_cross[0, 1]')\n"
        f"  - Absolute value: 'abs(<metric>)' or '<metric>_abs'"
    )


class OptunaObjective:
    """Objective function wrapping model training via local execution or Slurm batch submission."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        search_space: SearchSpace,
        study_run_dir: str,
        study_name: str = "optuna_study",
        executor: str = "local",
        slurm_header_file: Optional[str] = None,
        slurm_poll_interval: int = 10,
        metrics: Union[str, List[str]] = "val_loss",
        pruning_monitor: str = "val_loss",
        debug: bool = False,
        trials_subfolder: str = "trials",
        trial_output_to_file: bool = True,
    ):
        self.base_config = base_config
        self.search_space = search_space
        self.study_run_dir = os.path.abspath(study_run_dir)
        self.study_name = study_name
        self.executor = (executor or "local").lower()
        self.slurm_header_file = slurm_header_file
        self.slurm_poll_interval = max(int(slurm_poll_interval), 2)
        self.metrics = [metrics] if isinstance(metrics, str) else list(metrics)
        self.pruning_monitor = pruning_monitor
        self.debug = debug
        self.trials_subfolder = trials_subfolder
        self.trial_output_to_file = trial_output_to_file

    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        # 1. Sample hyperparameters
        overrides = self.search_space.sample(trial)

        # 2. Resolve final trial configuration
        trial_config = ConfigResolver.apply_overrides(self.base_config, overrides)
        _make_paths_absolute(trial_config)

        trial_outpath = os.path.join(self.study_run_dir, self.trials_subfolder)
        trial_folder_stem = "trial"
        trial_nexp = f"{trial.number:04d}"
        trial_run_dir = os.path.join(trial_outpath, f"{trial_folder_stem}_{trial_nexp}")

        trial_config["outpath"] = trial_outpath
        trial_config["outfolder"] = trial_folder_stem
        trial_config["nexp"] = trial_nexp
        trial_config["overwrite"] = True
        trial_config["output_to_file"] = self.trial_output_to_file
        if self.debug:
            trial_config["debug"] = True

        # 3. Execution Dispatcher
        if self.executor == "slurm":
            results = self._run_via_slurm(trial, trial_config, trial_run_dir)
        else:
            results = self._run_local(trial, trial_config)

        # 4. Record trial attributes and metrics in database
        trial.set_user_attr("run_dir", results.get("run_dir", trial_run_dir))
        trial.set_user_attr("best_checkpoint_path", results.get("best_checkpoint_path", ""))
        trial.set_user_attr("val_loss", float(results.get("val_loss", float("nan"))))

        test_results = results.get("test_results", {})
        if isinstance(test_results, dict):
            trial.set_user_attr("test_results", test_results)
            for k, v in test_results.items():
                if isinstance(v, (int, float)):
                    trial.set_user_attr(f"test_{k}", float(v))

        # 5. Extract objective metric(s)
        objective_values = [extract_metric(m, results, test_results) for m in self.metrics]
        if len(objective_values) == 1:
            return objective_values[0]
        return tuple(objective_values)

    def _run_local(self, trial: optuna.Trial, trial_config: Dict[str, Any]) -> Dict[str, Any]:
        """Runs training in-process on the local machine."""
        config, metargs = crb.prepare_from_config(
            trial_config,
            settings={"module": "trainer"},
            debug=self.debug
        )
        try:
            results = run_training_experiment(
                config=config,
                metargs=metargs,
                trial=trial if self.pruning_monitor != "none" else None,
                pruning_monitor=self.pruning_monitor,
            )
            return results
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            _log.error(f"Trial {trial.number} failed locally with error: {e}", exc_info=True)
            raise e

    def _run_via_slurm(self, trial: optuna.Trial, trial_config: Dict[str, Any], trial_run_dir: str) -> Dict[str, Any]:
        """Submits the trial to Slurm batch queue and monitors until completion."""
        trial_config_dir = os.path.join(self.study_run_dir, "trial_configs")
        os.makedirs(trial_config_dir, exist_ok=True)
        trial_config_file = os.path.join(trial_config_dir, f"trial_{trial.number:04d}.yaml")
        with open(trial_config_file, "w") as f:
            yaml.safe_dump(trial_config, f)

        # Staging directory for slurm scripts and logs
        slurm_job_dir = os.path.join(self.study_run_dir, "slurm_jobs", f"trial_{trial.number:04d}")
        os.makedirs(slurm_job_dir, exist_ok=True)

        # Initialize driver directly from header file using gslibs
        driver = slurm_driver.from_header_file(
            header_path=self.slurm_header_file,
            job_name=f"{self.study_name}_t{trial.number:04d}",
            run_dir=slurm_job_dir,
        )

        # Ensure job executes from working directory so relative dataset/trajectory paths resolve
        driver.add_command(f"cd {os.path.abspath(os.getcwd())}")
        driver.add_command(f"collective-encoder-train --config {os.path.abspath(trial_config_file)}")

        _log.info(f"Submitting Slurm job for Trial {trial.number} from {slurm_job_dir}")
        job_id = driver.submit_slurm()
        trial.set_user_attr("slurm_job_id", str(job_id))
        _log.info(f"Trial {trial.number} submitted to Slurm with Job ID {job_id}")

        # Wait for completion using gslibs lifecycle management
        final_status = driver.wait(poll_interval=self.slurm_poll_interval)
        trial.set_user_attr("slurm_final_status", final_status)
        _log.info(f"Trial {trial.number} (Slurm Job {job_id}) completed with status: {final_status}")

        if not driver.is_successful():
            raise RuntimeError(f"Slurm Job {job_id} for Trial {trial.number} finished with state: {final_status}")

        # Read results from metrics.yaml (or fallback to csv_logs)
        return self._read_trial_metrics(trial_run_dir)

    def _read_trial_metrics(self, trial_run_dir: str) -> Dict[str, Any]:
        """Reads trial metrics from metrics.yaml with automatic fallback to csv_logs."""
        metrics_path = os.path.join(trial_run_dir, "metrics.yaml")
        if os.path.isfile(metrics_path) and os.path.getsize(metrics_path) > 0:
            with open(metrics_path, "r") as f:
                results = yaml.safe_load(f) or {}
            if "val_loss" in results and results["val_loss"] is not None:
                return results

        # Robust Fallback: Extract best metrics from csv_logs
        best_val = float("inf")
        test_results = {}
        csv_dir = os.path.join(trial_run_dir, "csv_logs")
        if os.path.isdir(csv_dir):
            for root, _, files in os.walk(csv_dir):
                for fname in files:
                    if fname.endswith("metrics.csv"):
                        csv_file = os.path.join(root, fname)
                        try:
                            df = pd.read_csv(csv_file)
                            if "val_loss" in df.columns:
                                v_clean = df["val_loss"].dropna()
                                if not v_clean.empty:
                                    best_val = float(v_clean.min())
                            if "test_mae" in df.columns:
                                t_clean = df["test_mae"].dropna()
                                if not t_clean.empty:
                                    test_results["test_mae"] = float(t_clean.iloc[-1])
                        except Exception:
                            pass

        return {
            "val_loss": best_val,
            "run_dir": trial_run_dir,
            "best_checkpoint_path": os.path.join(trial_run_dir, "checkpoints", "best.ckpt"),
            "test_results": test_results,
        }

"""CLI runner and orchestrator for Optuna hyperparameter studies in collective_encoder."""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import optuna
import yaml
from gslibs.utils.common import recursive_update

import collective_encoder.ce_run_base as crb
from collective_encoder.hyperparameter_optimization.objective import OptunaObjective
from collective_encoder.hyperparameter_optimization.search_space import SearchSpace
from collective_encoder.hyperparameter_optimization.visualizer import (
    export_study_visualizations,
    launch_dashboard,
)

_log = logging.getLogger(__name__)

_SETTINGS = {
    "module": "optuna",
}


def _build_sampler(
    sampler_name: str,
    search_space: SearchSpace,
    sampler_kwargs: Dict[str, Any] = None
) -> optuna.samplers.BaseSampler:
    sampler_name = sampler_name.lower()
    kwargs = sampler_kwargs or {}
    if sampler_name in ("grid", "gridsampler") or search_space.is_grid_compatible():
        grid = search_space.get_grid_search_space()
        return optuna.samplers.GridSampler(search_space=grid, **kwargs)
    elif sampler_name in ("tpe", "tpesampler"):
        return optuna.samplers.TPESampler(**kwargs)
    elif sampler_name in ("cmaes", "cmaessampler"):
        return optuna.samplers.CmaEsSampler(**kwargs)
    elif sampler_name in ("botorch", "botorchsampler"):
        from optuna.integration import BoTorchSampler
        return BoTorchSampler(**kwargs)
    elif sampler_name in ("random", "randomsampler"):
        return optuna.samplers.RandomSampler(**kwargs)
    else:
        _log.warning(f"Unknown sampler: {sampler_name}, falling back to TPESampler")
        return optuna.samplers.TPESampler(**kwargs)


def _build_pruner(pruner_name: str, pruner_kwargs: Dict[str, Any] = None) -> optuna.pruners.BasePruner:
    pruner_name = (pruner_name or "none").lower()
    kwargs = pruner_kwargs or {}
    if pruner_name in ("none", "nop", "noppruner"):
        return optuna.pruners.NopPruner()
    elif pruner_name in ("hyperband", "hyperbandpruner"):
        return optuna.pruners.HyperbandPruner(**kwargs)
    elif pruner_name in ("median", "medianpruner"):
        return optuna.pruners.MedianPruner(**kwargs)
    elif pruner_name in ("percentile", "percentilepruner"):
        return optuna.pruners.PercentilePruner(**kwargs)
    else:
        return optuna.pruners.NopPruner()


MAXIMIZE_METRIC_KEYWORDS = {
    "mig", "beta_score", "beta_vae", "factor_score", "factor_vae",
    "dci", "modularity", "explicitness", "sap", "accuracy", "acc",
    "r2", "r_squared", "disentanglement", "score"
}


def infer_metric_direction(metric_name: str) -> str:
    """Infers optimization direction ('maximize' vs 'minimize') from metric name."""
    name_lower = str(metric_name).lower()
    for kw in MAXIMIZE_METRIC_KEYWORDS:
        if kw in name_lower:
            return "maximize"
    return "minimize"


class OptunaStudyRunner:
    """Manages Optuna study creation, optimization loop, and report generation."""

    def __init__(self, study_config: Dict[str, Any], metargs: Dict[str, Any]):
        self.config = study_config
        self.metargs = metargs
        self.study_run_dir = metargs["run_dir"]

        # Study name & SQLite storage in study directory
        self.study_name = self.config.get("study_name") or self.config.get("outfolder", "optuna_study")
        default_db_path = os.path.join(self.study_run_dir, "study.db")
        self.storage = self.config.get("storage") or f"sqlite:///{default_db_path}"

        # Study direction / objectives
        raw_metrics = self.config.get("metrics", "val_loss")
        self.metrics = [raw_metrics] if isinstance(raw_metrics, str) else list(raw_metrics)

        self.directions = self.config.get("directions", None)
        if self.directions is None:
            if "direction" in self.config:
                direction = self.config["direction"]
                self.directions = [direction] if isinstance(direction, str) else list(direction)
            else:
                self.directions = [infer_metric_direction(m) for m in self.metrics]

        # Base train config
        base_train_config_path = self.config.get("base_train_config", None)
        if not base_train_config_path or not os.path.isfile(base_train_config_path):
            raise FileNotFoundError(f"Base training config file not found: {base_train_config_path}")
        with open(base_train_config_path, "r") as f:
            self.base_train_config = yaml.safe_load(f)

        # Allow study config to override or attach test plotters
        if "test_plotters" in self.config:
            self.base_train_config["test_plotters"] = self.config["test_plotters"]
        if "test_plotter_type" in self.config:
            self.base_train_config["test_plotter_type"] = self.config["test_plotter_type"]
        if "test_plotter_args" in self.config:
            self.base_train_config["test_plotter_args"] = self.config["test_plotter_args"]

        # Search space
        space_dict = self.config.get("search_space", {})
        self.search_space = SearchSpace(space_dict)

        # Sampler & Pruner
        mode = self.config.get("mode", "").lower()
        sampler_setting = "grid" if mode == "grid" else self.config.get("sampler", "grid" if self.search_space.is_grid_compatible() else "tpe")
        pruner_setting = "none" if (mode == "grid" or sampler_setting == "grid") else self.config.get("pruner", "none")

        self.sampler = _build_sampler(
            sampler_setting,
            self.search_space,
            self.config.get("sampler_args", {})
        )
        self.pruner = _build_pruner(
            pruner_setting,
            self.config.get("pruner_args", {})
        )

        # Objective
        self.objective = OptunaObjective(
            base_config=self.base_train_config,
            search_space=self.search_space,
            study_run_dir=self.study_run_dir,
            study_name=self.study_name,
            executor=self.config.get("executor", "local"),
            slurm_header_file=self.config.get("slurm_header_file", None),
            slurm_poll_interval=self.config.get("slurm_poll_interval", 10),
            metrics=self.config.get("metrics", "val_loss"),
            pruning_monitor=self.config.get("pruning_monitor", "val_loss"),
            debug=self.config.get("debug", False),
            trials_subfolder=self.config.get("trials_subfolder", "trials"),
            trial_output_to_file=self.config.get("trial_output_to_file", True),
        )

    def create_or_load_study(self) -> optuna.Study:
        if self.config.get("overwrite", False):
            # Clean SQLite database file if overwriting
            if self.storage.startswith("sqlite:///"):
                db_path = self.storage.replace("sqlite:///", "")
                if os.path.isfile(db_path):
                    try:
                        os.remove(db_path)
                    except Exception:
                        pass
            try:
                optuna.delete_study(study_name=self.study_name, storage=self.storage)
            except Exception:
                pass

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            directions=self.directions,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=not self.config.get("overwrite", False),
        )
        return study

    def run(self, n_trials: Optional[int] = None, timeout: Optional[int] = None, n_jobs: int = 1) -> optuna.Study:
        study = self.create_or_load_study()

        # Compute default n_trials
        if n_trials is None:
            if "n_trials" in self.config and self.config["n_trials"] is not None:
                n_trials = self.config["n_trials"]
            elif self.search_space.is_grid_compatible():
                n_trials = self.search_space.total_combinations()
            else:
                n_trials = 20

        timeout = timeout or self.config.get("timeout", None)
        n_jobs = n_jobs or self.config.get("n_jobs", 1)

        callbacks = []
        if self.config.get("wandb", False):
            try:
                from optuna.integration.wandb import WeightsAndBiasesCallback
                wandb_kwargs = {
                    "metric_name": self.config.get("metrics", "val_loss"),
                    "project": self.config.get("wandb_project", "information-bottleneck"),
                    "entity": self.config.get("wandb_entity", None),
                }
                callbacks.append(WeightsAndBiasesCallback(**wandb_kwargs))
            except Exception as e:
                _log.warning(f"Could not initialize WeightsAndBiasesCallback: {e}")

        print("\n" + "=" * 65)
        print(f" Starting Optuna Study : {self.study_name}")
        print(f" Study Output Dir      : {self.study_run_dir}")
        print(f" Storage Database      : {self.storage}")
        print(f" Execution Backend     : {self.config.get('executor', 'local').upper()}")
        print(f" Target Trials / Grid  : {n_trials}")
        print(f" Objectives / Metrics  : {self.config.get('metrics', 'val_loss')} ({self.directions})")
        print(f" Sampler               : {type(self.sampler).__name__}")
        print(f" Parallel Workers      : {n_jobs}")
        print("=" * 65 + "\n")

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=callbacks,
        )

        self.print_summary(study)

        # Generate all static plots and summaries in {study_run_dir}/plots/ for explicitly specified metrics
        plots_output_dir = self.config.get("plot_dir") or os.path.join(self.study_run_dir, "plots")
        export_study_visualizations(
            study=study,
            output_dir=plots_output_dir,
            metrics=self.config.get("metrics", "val_loss"),
        )

        return study

    def print_summary(self, study: optuna.Study) -> None:
        print("\n" + "=" * 65)
        print(f" Optuna Study Completed : {study.study_name}")
        print(f" Total Trials Run       : {len(study.trials)}")

        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f" Complete Trials        : {len(complete_trials)}")

        if len(self.directions) == 1 and complete_trials:
            best_trial = study.best_trial
            print(f" Best Trial #{best_trial.number} Value: {best_trial.value:.6f}")
            print(" Best Hyperparameters   :")
            for k, v in best_trial.params.items():
                print(f"   - {k}: {v}")
        elif len(self.directions) > 1:
            print(" Best Pareto Trials (Multi-Objective):")
            for t in study.best_trials:
                print(f"   Trial #{t.number}: Values={t.values} Params={t.params}")
        print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hyperparameter Optimization studies using Optuna for collective_encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, type=str, help="Path to Optuna study YAML file")
    parser.add_argument("--base-train-config", type=str, default=None, help="Override path to baseline training YAML config")
    parser.add_argument("--executor", type=str, choices=["local", "slurm"], default=None, help="Execution backend: 'local' (in-process) or 'slurm' (HPC batch queue)")
    parser.add_argument("--slurm-header-file", type=str, default=None, help="Path to #SBATCH header and setup file for Slurm execution")
    parser.add_argument("--slurm-poll-interval", type=int, default=None, help="Seconds between Slurm status checks")
    parser.add_argument("--n-trials", "-n", type=int, default=None, help="Number of trials to execute on this worker")
    parser.add_argument("--timeout", type=int, default=None, help="Maximum execution time in seconds")
    parser.add_argument("--n-jobs", "-j", type=int, default=None, help="Number of parallel local workers / concurrent Slurm jobs")
    parser.add_argument("--storage", type=str, default=None, help="Override database storage URI (e.g. sqlite:///study.db)")
    parser.add_argument("--debug", action="store_true", help="Run fast 2-epoch debug trials")
    parser.add_argument("--plot-only", action="store_true", help="Generate/regenerate static plots without running new trials")
    parser.add_argument("--dashboard", action="store_true", help="Launch optuna-dashboard web UI")
    parser.add_argument("--port", type=int, default=8080, help="Port for optuna-dashboard")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load defaults and user config via crb
    default_config_path = crb.get_default_config_path(_SETTINGS)
    config = {}
    if os.path.isfile(default_config_path):
        with open(default_config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    with open(args.config, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    recursive_update(config, user_cfg)

    if args.base_train_config:
        config["base_train_config"] = args.base_train_config
    if args.executor:
        config["executor"] = args.executor
    if args.slurm_header_file:
        config["slurm_header_file"] = args.slurm_header_file
    if args.slurm_poll_interval is not None:
        config["slurm_poll_interval"] = args.slurm_poll_interval
    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs
    if args.storage:
        config["storage"] = args.storage
    if args.debug:
        config["debug"] = True

    # If plot-only or dashboard, load study from existing run_dir or storage
    if args.plot_only:
        study_name = config.get("study_name") or config.get("outfolder", "optuna_study")
        study_run_dir = os.path.join(config.get("outpath", "./train_runs/hpo"), f"{config.get('outfolder', 'sweep')}_{config.get('nexp', 1)}")
        storage = config.get("storage") or f"sqlite:///{study_run_dir}/study.db"
        study = optuna.load_study(study_name=study_name, storage=storage)
        plots_output_dir = config.get("plot_dir") or os.path.join(study_run_dir, "plots")
        export_study_visualizations(study, plots_output_dir)
        print(f"Plots and reports regenerated in: {plots_output_dir}")
        return

    if args.dashboard:
        study_run_dir = os.path.join(config.get("outpath", "./train_runs/hpo"), f"{config.get('outfolder', 'sweep')}_{config.get('nexp', 1)}")
        storage = config.get("storage") or f"sqlite:///{study_run_dir}/study.db"
        launch_dashboard(storage, port=args.port)
        return

    # Clean study directory if overwrite is True
    if config.get("overwrite", False):
        import shutil
        target_stem = config.get("outfolder", "sweep")
        target_nexp = config.get("nexp", 1)
        target_path = config.get("outpath", "./train_runs/hpo")
        target_dir = os.path.join(target_path, f"{target_stem}_{target_nexp}")
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
            if os.path.isdir(target_dir):
                for item in os.listdir(target_dir):
                    item_path = os.path.join(target_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        try:
                            os.remove(item_path)
                        except Exception:
                            pass

    # Normal Study Execution: Prepare study run_dir via crb
    config, metargs = crb.prepare_from_config(config, settings=_SETTINGS, config_path=args.config, debug=args.debug)

    runner = OptunaStudyRunner(config, metargs)
    runner.run(n_trials=args.n_trials, timeout=args.timeout, n_jobs=config.get("n_jobs", 1))


if __name__ == "__main__":
    main()

import importlib.util
import os
import yaml
import pytest

import collective_encoder.ce_run_base as crb
from collective_encoder.trainer import run_training_experiment


def _load_config_resolver():
    resolver_path = os.path.join(
        os.path.dirname(crb.__file__),
        "hyperparameter_optimization",
        "resolver.py",
    )
    spec = importlib.util.spec_from_file_location("_ce_hpo_resolver", resolver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConfigResolver


@pytest.mark.unit
def test_hpo_config_resolver():
    ConfigResolver = _load_config_resolver()
    base_cfg = {
        "lrate": 1e-3,
        "weight_decay": 0.0,
        "datamodule_args": {"batch_size": 32, "val_batch_size": 32},
        "network_args": {
            "network": [20, 16, 8, 3],
            "encoder_network": [20, 16, 8, 3],
            "beta": 1.0,
        },
    }
    overrides = {
        "latent_dim": 2,
        "hidden_layers": "[20, 12]",
        "beta": 4.0,
        "lr": "0.0005",
        "batch_size": 16,
        "kld_max_value": 0.5,
        "network_args.activation": "elu",
    }
    resolved = ConfigResolver.apply_overrides(base_cfg, overrides)
    assert resolved["lrate"] == 0.0005
    assert resolved["datamodule_args"]["batch_size"] == 16
    assert resolved["network_args"]["network"] == [20, 12, 2]
    assert resolved["network_args"]["encoder_network"] == [20, 12, 2]
    assert resolved["network_args"]["beta"] == 4.0
    assert resolved["network_args"]["activation"] == "elu"
    assert resolved["network_args"]["kld_args"]["kld_max_scheduler_args"]["value"] == 0.5

    changes = ConfigResolver.diff(base_cfg, resolved)
    assert "lrate" in changes
    assert changes["lrate"] == (1e-3, 0.0005)


@pytest.mark.unit
def test_hpo_search_space_grid_and_continuous():
    optuna = pytest.importorskip("optuna")
    from collective_encoder.hyperparameter_optimization.search_space import SearchSpace

    grid_space = SearchSpace(
        {
            "latent_dim": [2, 3],
            "beta": {"type": "categorical", "choices": [0.5, 1.0]},
            "depth": {"type": "int", "low": 1, "high": 2, "step": 1},
        }
    )
    assert grid_space.is_grid_compatible() is True
    assert grid_space.total_combinations() == 2 * 2 * 2

    cont_space = SearchSpace(
        {
            "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "dropout": {"type": "uniform", "low": 0.0, "high": 0.5},
        }
    )
    assert cont_space.is_grid_compatible() is False
    assert cont_space.total_combinations() == 0

    study = optuna.create_study()
    trial = study.ask()
    sampled = cont_space.sample(trial)
    assert 1e-4 <= sampled["lr"] <= 1e-2
    assert 0.0 <= sampled["dropout"] <= 0.5


@pytest.mark.unit
def test_hpo_extract_metric_and_direction():
    pytest.importorskip("optuna")
    from collective_encoder.hyperparameter_optimization.objective import extract_metric
    from collective_encoder.hyperparameter_optimization.runner import infer_metric_direction

    assert infer_metric_direction("val_loss") == "minimize"
    assert infer_metric_direction("DIS_mig") == "maximize"
    assert infer_metric_direction("beta_score") == "maximize"

    results = {"val_loss": 0.25}
    test_results = {
        "mig_score": 0.82,
        "C_cross_matrix": {
            "x_labels": ["LD_1", "LD_2"],
            "y_labels": ["phi", "psi"],
            "matrix": [[-0.9, 0.1], [0.2, 0.85]],
        },
    }
    assert extract_metric("val_loss", results, test_results) == 0.25
    assert extract_metric("mig_score", results, test_results) == 0.82
    assert extract_metric("C_cross[LD_1, phi]", results, test_results) == -0.9
    assert extract_metric("abs(C_cross[LD_1, phi])", results, test_results) == 0.9
    assert extract_metric("C_cross[0, 1]", results, test_results) == 0.1

    with pytest.raises(KeyError, match="not found"):
        extract_metric("nonexistent_metric", results, test_results)


@pytest.mark.integration
def test_end_to_end_training_experiment(trajectory_files, tmp_path):
    pdb_path, xtc_path = trajectory_files
    outpath = str(tmp_path / "runs")

    train_cfg = {
        "outpath": outpath,
        "outfolder": "e2e_sae",
        "nexp": 1,
        "overwrite": True,
        "output_to_file": False,
        "seed": 42,
        "nepochs": 1,
        "nogpu": True,
        "datamodule_type": "COORDINATES",
        "datamodule_args": {
            "datareader_type": "COMPRESSED_TRAJECTORY",
            "datareader_args": {
                "topology_file": pdb_path,
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
                "parallel": False,
            },
            "dataset_type": "DISTANCES",
            "train_size": 8,
            "batch_size": 4,
            "validation_size": 4,
            "val_batch_size": 4,
            "test_size": 4,
            "test_batch_size": 4,
            "num_workers": 0,
        },
        "network_type": "sAE",
        "network_args": {
            "network": [8, 2],
            "lrate": 1e-3,
        },
        "test_plotters": [],
    }

    config, metargs = crb.prepare_from_config(
        train_cfg,
        settings={"module": "trainer"},
    )
    results = run_training_experiment(config=config, metargs=metargs)
    assert "val_loss" in results
    assert os.path.isdir(results["run_dir"])
    assert os.path.isfile(os.path.join(results["run_dir"], "metrics.yaml"))


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bug in collective_encoder/hyperparameter_optimization/objective.py:L449: "
        "_read_trial_metrics fallback attempts dict + dict ('{...} + remaining_metrics'), raising TypeError"
    ),
)
def test_optuna_objective_read_trial_metrics_csv_fallback(tmp_path):
    pytest.importorskip("optuna")
    from collective_encoder.hyperparameter_optimization.objective import OptunaObjective
    from collective_encoder.hyperparameter_optimization.search_space import SearchSpace

    obj = OptunaObjective(
        base_config={},
        search_space=SearchSpace({}),
        study_run_dir=str(tmp_path),
    )
    trial_dir = tmp_path / "trial_0000"
    trial_dir.mkdir(parents=True, exist_ok=True)
    obj._read_trial_metrics(str(trial_dir))


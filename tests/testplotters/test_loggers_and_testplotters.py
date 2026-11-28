import os
import numpy as np
import pandas as pd
import pytest

from collective_encoder.loggers.csv_plot_logger import CSVPlotLogger, plot_metrics_csv
from collective_encoder.testplotters.disentanglement import DisentanglementPlotter


@pytest.mark.unit
def test_csv_plot_logger_and_plot_metrics_csv(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    df = pd.DataFrame(
        {
            "epoch": [0, 0, 1, 1, 2, 2],
            "step": [1, 2, 3, 4, 5, 6],
            "train_loss": [1.2, np.nan, 0.8, np.nan, 0.5, np.nan],
            "val_loss": [np.nan, 1.1, np.nan, 0.7, np.nan, 0.4],
            "train_rec_loss": [1.0, np.nan, 0.6, np.nan, 0.3, np.nan],
            "val_rec_loss": [np.nan, 0.9, np.nan, 0.5, np.nan, 0.25],
        }
    )
    df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "plots"
    saved = plot_metrics_csv(str(csv_file), output_dir=str(out_dir))
    assert len(saved) >= 2
    for p in saved:
        assert os.path.isfile(p)

    # CSVPlotLogger wrapper instantiates cleanly
    logger = CSVPlotLogger(save_dir=str(tmp_path), name="csv_exp")
    assert logger.name == "csv_exp"


@pytest.mark.unit
def test_disentanglement_plotter_with_plots_enabled(metargs):
    rng = np.random.default_rng(42)
    f1 = np.repeat(np.linspace(-1.0, 1.0, 10), 10)
    f2 = np.tile(np.linspace(-1.0, 1.0, 10), 10)
    z = np.column_stack([f1 + rng.normal(0, 0.02, 100), f2 + rng.normal(0, 0.02, 100)])
    labels = {"f1": f1, "f2": f2}

    plotter = DisentanglementPlotter(
        args={
            "name": "DIS",
            "metrics": ["mig", "sap"],
            "num_bins": 5,
        },
        **metargs,
    )
    plotter.add_batch(data=z, latent=z, pred=z, labels=labels, meta={})
    plotter.finish()
    metrics = plotter.get_metrics()
    assert "DIS_mig" in metrics
    assert "DIS_sap" in metrics


@pytest.mark.unit
def test_disentanglement_plotter_scores_when_heatmap_plots_disabled(metargs):
    """
    When plot_confusion_matrix, plot_importance_matrix, plot_mi_matrix, and plot_sap_matrix
    are False (except modularity which always plots), MIG, SAP, DCI, Beta, and Factor metrics
    compute valid disentanglement scores in [0, 1].
    """
    rng = np.random.default_rng(42)
    f1 = np.repeat(np.linspace(-1.0, 1.0, 15), 10)
    f2 = np.tile(np.linspace(-1.0, 1.0, 10), 15)
    z = np.column_stack([f1 + rng.normal(0, 0.01, 150), f2 + rng.normal(0, 0.01, 150)])
    labels = {"f1": f1, "f2": f2}

    plotter = DisentanglementPlotter(
        args={
            "name": "DIS",
            "metrics": ["beta", "factor", "dci", "mig", "sap"],
            "num_models": 2,
            "num_bins": 5,
            "plot_confusion_matrix": False,
            "plot_importance_matrix": False,
            "plot_mi_matrix": False,
            "plot_sap_matrix": False,
        },
        **metargs,
    )
    plotter.add_batch(data=z, latent=z, pred=z, labels=labels, meta={})
    plotter.finish()
    metrics = plotter.get_metrics()
    for key in ["DIS_beta", "DIS_factor", "DIS_dci", "DIS_mig", "DIS_sap", "DIS_mean"]:
        assert key in metrics, f"Expected {key} in {list(metrics.keys())}"
        assert 0.0 <= metrics[key] <= 1.05

import logging
import os
import shutil
from typing import Any, Dict, Optional
import yaml

_log = logging.getLogger(__name__)

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from collective_encoder.loggers import CSVPlotLogger
import collective_encoder.ce_run_base as crb

_SETTINGS = {
    'module': 'trainer',
}


def _setup_logger(config: dict, run_dir: str):
    """Set up the appropriate PyTorch Lightning logger based on config."""
    logger_name = config.get('logger_name', None)
    if logger_name is None:
        if config.get('wandb', False):
            logger_name = 'wandb'
        else:
            logger_name = 'csv'

    if logger_name == 'wandb':
        args = {
            'save_dir': run_dir,
            'name': run_dir.strip(".").strip("/").replace("/", "_"),
            'log_model': False,
        }
        logger_args = config.get('logger_args', {})
        if 'project' not in logger_args and 'wandb_project' in config:
            logger_args['project'] = config['wandb_project']
        if 'entity' not in logger_args and 'wandb_entity' in config:
            logger_args['entity'] = config['wandb_entity']

        args.update(logger_args)
        if 'project' not in args or args['project'] in [None, 'None']:
            raise ValueError("Wandb logger requires 'project' argument in 'logger_args' or 'wandb_project'.")
        if 'entity' not in args or args['entity'] in [None, 'None']:
            raise ValueError("Wandb logger requires 'entity' argument in 'logger_args' or 'wandb_entity'.")
        return WandbLogger(**args)
    else:
        return CSVPlotLogger(
            save_dir=run_dir,
            name="csv_logs",
            version="version_0",
            **config.get('logger_args', {})
        )


def run_training_experiment(
    config: Dict[str, Any],
    metargs: Dict[str, Any],
    trial: Optional[Any] = None,
    pruning_monitor: str = 'val_loss',
) -> Dict[str, Any]:
    """
    Run a training experiment using prepared config and metargs dictionaries.
    
    Args:
        config: Training configuration dictionary.
        metargs: Metadata dictionary containing run_dir, verbose, etc.
        trial: Optional Optuna trial object for pruning callback.
        pruning_monitor: Metric name to monitor for Optuna pruning.

    Returns:
        Dictionary containing experiment results and metrics (val_loss, best_checkpoint_path, run_dir, test_results).
    """
    dm = crb.load_datamodule(config, metargs)
    model = crb.load_model(config, metargs, dm)
    run_dir = metargs['run_dir']

    test_results = {}
    best_val_loss = float('inf')
    best_checkpoint_path = ""

    ##################################
    # Training / Fitting the Model
    ##################################
    if isinstance(model, pl.LightningModule):
        trainargs = {
            "max_epochs": config.get('nepochs', 50),
            "log_every_n_steps": 1,
            "default_root_dir": run_dir,
        }
        if not config.get('nogpu', False):
            trainargs["accelerator"] = 'auto'
            trainargs["devices"] = 'auto'
        else:
            trainargs["accelerator"] = 'cpu'

        trainargs["logger"] = _setup_logger(config, run_dir)

        ## PL Callbacks
        callbacks = []
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

        # Early stopping
        if config.get('early_stopping', True):
            early_stop_callback = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.get('early_stopping_args', {}).get('patience', 100),
                min_delta=config.get('early_stopping_args', {}).get('min_delta', 1e-8),
                verbose=True,
                mode='min'
            )
            callbacks.append(early_stop_callback)

        # Optuna pruning callback if trial is provided
        if trial is not None:
            try:
                from optuna.integration import PyTorchLightningPruningCallback
                callbacks.append(PyTorchLightningPruningCallback(trial, monitor=pruning_monitor))
            except Exception as e:
                _log.warning(f"Could not attach PyTorchLightningPruningCallback: {e}")

        model_name = config.get('network_type', config.get('model_type', 'model'))
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join(run_dir, 'checkpoints'),
            filename=model_name + '-{epoch:02d}-{val_loss:.6f}',
            save_top_k=1,
            mode='min',
        )
        callbacks.append(checkpoint_callback)
        trainargs["callbacks"] = callbacks

        trainer = pl.Trainer(**trainargs)

        nepochs = config.get('nepochs', 50)
        if nepochs > 0:
            _log.info("Starting training for %d epochs...", nepochs)
            trainer.fit(model, datamodule=dm)
            _log.info("Training completed.")
            if config.get('wandb', False):
                try:
                    import wandb
                    wandb.finish()
                except Exception:
                    pass

        if nepochs == 0 and 'load_model' not in config and 'load_network' not in config:
            _log.warning("Both nepochs and load_model are not set. Nothing to do.")

        best_checkpoint_path = checkpoint_callback.best_model_path
        if best_checkpoint_path != "" and os.path.exists(best_checkpoint_path):
            shutil.copy(best_checkpoint_path, os.path.join(os.path.dirname(best_checkpoint_path), "best.ckpt"))
            if checkpoint_callback.best_model_score is not None:
                score = checkpoint_callback.best_model_score
                best_val_loss = float(score.item() if hasattr(score, 'item') else score)
        _log.info(f"Best model saved at: {best_checkpoint_path} with val_loss: {best_val_loss}")

        ##################################
        # Testing the NN
        ##################################
        if config.get('test_plotters', False) and isinstance(config['test_plotters'], list):
            for tp in config['test_plotters']:
                if 'tester_type' in tp:
                    model.add_test_plotter(tp['tester_type'], tp.get('tester_args', None))
        elif config.get('test_plotter_type', False):
            model.add_test_plotter(config['test_plotter_type'], config.get('test_plotter_args', None))

        _log.info("Starting testing...")
        test_out = trainer.test(model, datamodule=dm)
        if test_out and len(test_out) > 0:
            test_results = dict(test_out[0])
        plotter_metrics = model.get_test_plotter_metrics() if hasattr(model, "get_test_plotter_metrics") else {}
        test_results.update(plotter_metrics)
        _log.info(f"Testing completed. Test metrics: {list(test_results.keys())}")

    else:
        # Analytical non-gradient model (e.g. PCA, ICA)
        _log.info(f"Fitting {model.__class__.__name__} analytically on training data...")
        model.fit(datamodule=dm)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
        model.save_checkpoint(best_ckpt)
        best_checkpoint_path = best_ckpt
        _log.info(f"Fitted model saved at: {best_ckpt}")

        ##################################
        # Testing the Model
        ##################################
        if config.get('test_plotters', False) and isinstance(config['test_plotters'], list):
            for tp in config['test_plotters']:
                if 'tester_type' in tp:
                    model.add_test_plotter(tp['tester_type'], tp.get('tester_args', None))
        elif config.get('test_plotter_type', False):
            model.add_test_plotter(config['test_plotter_type'], config.get('test_plotter_args', None))

        _log.info("Starting testing...")
        test_out = model.test(datamodule=dm)
        if isinstance(test_out, dict):
            test_results = dict(test_out)
        plotter_metrics = model.get_test_plotter_metrics() if hasattr(model, "get_test_plotter_metrics") else {}
        test_results.update(plotter_metrics)
        _log.info(f"Testing completed. Test metrics: {list(test_results.keys())}")

    #####################################
    # Save metatomic model
    #####################################
    if config.get('save_metatomic', False):
        try:
            from collective_encoder.mtomic.wrapper import MetatomicCV
            from metatomic.torch import (
                AtomisticModel,
                ModelCapabilities,
                ModelMetadata,
                ModelOutput,
                System,
                ModelEvaluationOptions,
            )
            from metatensor.torch import Labels
        except ImportError:
            raise ImportError("metatomic is not installed. Please install it with `pip install metatomic`")

        dataprocessor = dm.get_dataset().get_metatomic_dataprocessor()
        metamodel = model.get_metatomic_model()
        metatomic_model = MetatomicCV(dataprocessor, metamodel)
        metadata = ModelMetadata(
            name=config.get('metatomic_metadata', {}).get('name', 'unknown'),
            description=config.get('metatomic_metadata', {}).get('description', 'unknown'),
            authors=config.get('metatomic_metadata', {}).get('authors', []),
            references=config.get('metatomic_metadata', {}).get('references', {}),
        )
        capabilities = ModelCapabilities(
            outputs=metamodel.get_metatomic_outputs(),
            atomic_types=dataprocessor.get_atomic_types(),
            interaction_range=dataprocessor.get_interaction_range(),
            length_unit=dataprocessor.get_length_unit(),
            supported_devices=["cpu", "cuda"],
            dtype="float64",
        )
        metatomic_module = AtomisticModel(
            module=metatomic_model.eval(),
            metadata=metadata,
            capabilities=capabilities,
        )
        fake_systems = dm.get_dataset().get_fake_systems()
        fake_options = ModelEvaluationOptions(
            length_unit=dataprocessor.get_length_unit(),
            outputs=metamodel.get_metatomic_outputs(),
            selected_atoms=None,
        )
        try:
            print("Running sanity check of the metatomic model...")
            with torch.no_grad():
                output = metatomic_module(fake_systems, fake_options, False)
        except Exception as e:
            raise RuntimeError("metatomic model failed the sanity check: " + str(e))

        print("metatomic model passed the sanity check.\nSerializing the model...")
        metatomic_model_file = os.path.join(run_dir, "metatomic_model.pt")
        metatomic_extension_directory = os.path.join(run_dir, "metatomic_extensions")
        metatomic_module.save(
            metatomic_model_file,
            collect_extensions=metatomic_extension_directory
        )
        print(f"@@ metatomic model saved as: {metatomic_model_file}")

    def _to_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if hasattr(val, "item"):
            try:
                return val.item()
            except Exception:
                pass
        if isinstance(val, dict):
            return {k: _to_serializable(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_to_serializable(x) for x in val]
        return val

    results = {
        'val_loss': _to_serializable(best_val_loss),
        'best_checkpoint_path': str(best_checkpoint_path) if best_checkpoint_path else "",
        'run_dir': str(run_dir),
        'test_results': _to_serializable(test_results),
    }

    try:
        metrics_file = os.path.join(run_dir, "metrics.yaml")
        with open(metrics_file, "w") as f:
            yaml.safe_dump(results, f)
    except Exception as e:
        _log.warning(f"Could not save metrics.yaml to {run_dir}: {e}")

    return results


def train():
    """Main CLI entry point for collective-encoder-train."""
    config, metargs = crb.prepare(_SETTINGS)
    run_training_experiment(config, metargs)


def main():
    """Entry point for trainer module."""
    train()


if __name__ == "__main__":
    main()






import logging
import os
import wandb
import shutil

_log = logging.getLogger(__name__)

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import collective_encoder.ce_run_base as crb

_SETTINGS = {
    'module': 'trainer',
}

def train():
    """Train a collective encoder model based on the provided configuration."""
    
    config, metargs = crb.prepare(_SETTINGS)
    dm = crb.load_datamodule(config, metargs)
    model = crb.load_model(config, metargs, dm)
    run_dir = metargs['run_dir']

    ##################################
    # Training the NN
    ##################################
    trainargs = {"max_epochs" : config['nepochs'],
                 "log_every_n_steps" : 1,
                 "default_root_dir" : run_dir}
    if not config.get('nogpu', False):
        trainargs["accelerator"] = 'auto'
        trainargs["devices"] = 'auto'
    
    ## External logging
    if config.get('wandb', False):
        wandb_logger = WandbLogger(project=config['wandb_project'],
                                 entity=config['wandb_entity'],
                                 save_dir=run_dir,
                                 name=run_dir.strip(".").strip("/").replace("/", "_"),
                                 log_model=False,)
        # wandb_logger.watch(model, log_graph=True)
        trainargs["logger"] = wandb_logger

    ## PL Callbacks
    callbacks = []
    # Learning rate monitor
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

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=run_dir + '/checkpoints',
        filename=config['network_type'] + '-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1,
        mode='min',
    )
    callbacks.append(checkpoint_callback)

    trainargs["callbacks"] = callbacks
    # trainargs["num_sanity_val_steps"] = 0

    # trainargs["gradient_clip_val"] = 0.5
    # trainargs["gradient_clip_algorithm"] = "norm"

    trainer = pl.Trainer(**trainargs)

    if config['nepochs'] > 0:
        _log.info("Starting training for %d epochs...", config['nepochs'])
        trainer.fit(model, datamodule=dm)
        _log.info("Training completed.")
        if config.get('wandb', False):
            wandb.finish()

    if config['nepochs'] == 0 and 'load_model' not in config:
        _log.warning("Both nepochs and load_model are not set. Nothing to do.")

    # Save the best model checkpoint as best.ckpt
    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path != "":
        shutil.copy(best_checkpoint_path, os.path.dirname(best_checkpoint_path) + "/best.ckpt")
    _log.info(f"Best model saved at: {best_checkpoint_path}")
    
    
    ##################################
    # Testing the NN
    ##################################
    if config.get('test_plotter_type', False):
        model.add_test_plotter(config['test_plotter_type'], config.get('test_plotter_args', None))
    _log.info("Starting testing...")
    trainer.test(model, datamodule=dm)
    _log.info("Testing completed.")

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
        # FIXME: Remove wandb hooks to avoid issues during serialization
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
        ##################################
        # Sanity check of the model
        ##################################
        fake_systems = dm.get_dataset().get_fake_systems()
        fake_options = ModelEvaluationOptions(
            length_unit=dataprocessor.get_length_unit(),
            outputs=metamodel.get_metatomic_outputs(),
            selected_atoms=None,
            # selected_atoms=Labels(
            #     ['system', 'atom'], torch.tensor([0, 512]).reshape(-1, 2)
            # ),
        )
            
        # Run inference
        try:
            print("Running sanity check of the metatomic model...")
            with torch.no_grad():
                output = metatomic_module(fake_systems, fake_options, False)
        except Exception as e:
            raise RuntimeError("metatomic model failed the sanity check: "+str(e))

        print("metatomic model passed the sanity check.\nSerializing the model...")
        metatomic_model_file = os.path.join(run_dir, "metatomic_model.pt")
        metatomic_extension_directory = os.path.join(run_dir, "metatomic_extensions")
        metatomic_module.save(metatomic_model_file,
                            collect_extensions=metatomic_extension_directory)

        print(f"@@ metatomic model saved as: {metatomic_model_file}")

def main():
    """Main entry point for the collective encoder training."""
    train()

if __name__ == "__main__":
    main()






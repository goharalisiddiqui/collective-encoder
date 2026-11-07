import pytest
import torch
import pytorch_lightning as pl

from collective_encoder import TENSOR_PRECISION
from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.models import (
    PCAModel,
    ICAModel,
    TICAModel,
    AE,
    sAE,
    VAE,
    sVAE,
    DVAE,
    sDVAE,
    EDVAE,
    BondGraphEncoderDecoder,
)


@pytest.fixture
def distances_datamodule(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files
    return CoordinatesDataModule(
        args={
            "datareader_type": "COMPRESSED_TRAJECTORY",
            "datareader_args": {
                "topology_file": pdb_path,
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
                "parallel": False,
            },
            "dataset_type": "DISTANCES",
            "labeler_type": "DISTANCE",
            "labeler_args": {"selections": ["index 0 or index 1"]},
            "train_size": 8,
            "batch_size": 4,
            "validation_size": 4,
            "val_batch_size": 4,
            "test_size": 4,
            "test_batch_size": 4,
            "sequential": True,
            "num_workers": 0,
        },
        **metargs,
    )


@pytest.fixture
def bondgraph_datamodule(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files
    return CoordinatesDataModule(
        args={
            "datareader_type": "COMPRESSED_TRAJECTORY",
            "datareader_args": {
                "topology_file": pdb_path,
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
                "parallel": False,
            },
            "dataset_type": "BONDGRAPH",
            "dataset_args": {"precompute_graphs": False, "parallel": False},
            "train_size": 8,
            "batch_size": 4,
            "validation_size": 4,
            "val_batch_size": 4,
            "test_size": 4,
            "test_batch_size": 4,
            "sequential": True,
            "num_workers": 0,
        },
        **metargs,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls,extra_args",
    [
        (PCAModel, {"latent_dim": 2, "center": True}),
        (ICAModel, {"latent_dim": 2, "max_iter": 100}),
        (TICAModel, {"latent_dim": 2, "lag": 1}),
    ],
)
def test_classical_models_pca_ica_tica(distances_datamodule, tmp_path, metargs, model_cls, extra_args):
    args = model_cls.extract_args_from_datamodule(distances_datamodule, extra_args.copy())
    model = model_cls(args=args, **metargs)
    model.fit(datamodule=distances_datamodule)
    model.test(datamodule=distances_datamodule)

    batch = next(iter(distances_datamodule.train_dataloader()))
    x = batch[0]
    z, _ = model.encoder(x)
    x_rec, _ = model.decoder(z)
    assert z.shape == (4, 2)
    assert x_rec.shape == x.shape

    ckpt_path = str(tmp_path / f"{model_cls.__name__}.ckpt")
    model.save_checkpoint(ckpt_path)
    reloaded = model_cls.load_from_checkpoint(ckpt_path, **metargs)
    z_rel, _ = reloaded.encoder(x)
    torch.testing.assert_close(z, z_rel)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls,net_args",
    [
        (AE, {"encoder_network": [8, 2], "decoder_network": [8]}),
        (sAE, {"network": [8, 2]}),
        (VAE, {"encoder_network": [8, 2], "decoder_network": [8], "beta": 0.5}),
        (sVAE, {"network": [8, 2], "beta": 0.5}),
        (DVAE, {"encoder_network": [8, 2], "decoder_network": [8], "beta": 0.5}),
        (sDVAE, {"network": [8, 2], "beta": 0.5}),
    ],
)
def test_flat_neural_net_models_ae_vae_dvae(distances_datamodule, metargs, model_cls, net_args):
    raw_args = {
        **net_args,
        "lrate": 1e-3,
        "batch_norm": False,
        "activation": "relu",
    }
    args = model_cls.extract_args_from_datamodule(distances_datamodule, raw_args)
    model = model_cls(args=args, **metargs)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    
    trainer.fit(model, datamodule=distances_datamodule)
    trainer.validate(model, datamodule=distances_datamodule)
    trainer.test(model, datamodule=distances_datamodule)

    batch = next(iter(distances_datamodule.train_dataloader()))
    out, z, meta = model(batch[0])
    assert z.shape == (4, 2)
    assert out.shape == (4, 15)


@pytest.mark.unit
def test_edvae_on_1d_distances(distances_datamodule, metargs):
    raw_args = {
        "encoder_network": [8, 2],
        "decoder_network": [8],
        "embedding_type": "flatten",
        "lrate": 1e-3,
    }
    args = EDVAE.extract_args_from_datamodule(distances_datamodule, raw_args)
    model = EDVAE(args=args, **metargs)
    batch = next(iter(distances_datamodule.train_dataloader()))
    out, z, meta = model(batch[0])
    assert z.shape == (4, 2)
    assert out.shape == (4, 15)


@pytest.mark.unit
def test_edvae_on_3d_positions_shape(metargs):
    model = EDVAE(
        args={
            "encoder_network": [8, 2],
            "decoder_network": [8],
            "datapoint_shape": (6, 3),
            "dataset_type": "POSITIONS",
            "embedding_type": "flatten",
            "lrate": 1e-3,
        },
        **metargs,
    )
    x = torch.randn(4, 6, 3, dtype=TENSOR_PRECISION)
    out, z, meta = model(x)
    assert out.shape == (4, 6, 3)


@pytest.mark.unit
def test_bge_on_bondgraph(bondgraph_datamodule, metargs):
    raw_args = {
        "encoder_args": {
            "latent_dim": 2,
            "num_layers": 1,
            "hidden_dim": 8,
            "heads": 2,
        },
        "decoder_args": {
            "template_khop": 2,
            "num_layers": 1,
            "hidden_dim": 8,
            "heads": 2,
        },
        "lrate": 1e-3,
        "normIn": True,
    }
    args = BondGraphEncoderDecoder.extract_args_from_datamodule(bondgraph_datamodule, raw_args)
    model = BondGraphEncoderDecoder(args=args, **metargs)
    # Not attached to a Trainer, so pass the datamodule explicitly for normalization stats
    model.set_norm(datamodule=bondgraph_datamodule)
    batch = next(iter(bondgraph_datamodule.train_dataloader()))
    out, z, meta = model(batch)
    assert z.shape == (4, 2)
    assert isinstance(out, dict)

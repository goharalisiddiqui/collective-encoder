import os
import pytest
import numpy as np
import torch

import collective_encoder.ce_run_base as crb
from collective_encoder.datamodules.base import BaseDataModule
from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.datamodules.colvars import ColvarsDataModule


@pytest.mark.unit
def test_coordinates_datamodule_random_and_sequential_splits(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files

    # 1. Random split with train + val + test + predict using DISTANCES
    dmod_rand = CoordinatesDataModule(
        args={
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
            "val_batch_size": 2,
            "test_size": 4,
            "test_batch_size": 2,
            "predict_size": 2,
            "predict_batch_size": 2,
            "sequential": False,
            "num_workers": 0,
        },
        **metargs,
    )

    # Disjointness of random split indices
    s_train = set(dmod_rand.train_indices)
    s_val = set(dmod_rand.val_indices)
    s_test = set(dmod_rand.test_indices)
    s_pred = set(dmod_rand.predict_indices)
    assert len(s_train) == 8
    assert len(s_val) == 4
    assert len(s_test) == 4
    assert len(s_pred) == 2
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_train.isdisjoint(s_pred)
    assert s_val.isdisjoint(s_test)
    assert s_val.isdisjoint(s_pred)
    assert s_test.isdisjoint(s_pred)

    # Check coordinate metadata accessors
    assert len(dmod_rand.get_atns()) == 6
    assert len(dmod_rand.get_element_symbols()) == 6
    assert len(dmod_rand.get_bond_indices()) == 5
    assert dmod_rand.get_datapoint_shape() == (15,)

    # Check dataloaders and batch shapes
    train_batch = next(iter(dmod_rand.train_dataloader()))
    val_batch = next(iter(dmod_rand.val_dataloader()))
    test_batch = next(iter(dmod_rand.test_dataloader()))
    pred_batch = next(iter(dmod_rand.predict_dataloader()))
    full_batch = dmod_rand.get_full_batch()

    assert train_batch[0].shape == (4, 15)
    assert val_batch[0].shape == (2, 15)
    assert test_batch[0].shape == (2, 15)
    assert pred_batch[0].shape == (2, 15)
    assert full_batch[0].shape == (18, 15)

    # 2. Sequential split (predict_size=0)
    dmod_seq = CoordinatesDataModule(
        args={
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
            "test_size": 4,
            "sequential": True,
            "num_workers": 0,
        },
        **metargs,
    )
    assert len(dmod_seq.train_indices) == 8
    assert len(dmod_seq.val_indices) == 4
    assert len(dmod_seq.test_indices) == 4
    assert dmod_seq.val_indices[0] == dmod_seq.train_indices[-1] + 1
    assert dmod_seq.test_indices[0] == dmod_seq.val_indices[-1] + 1
    assert dmod_seq.get_datapoint_shape() == (15,)


@pytest.mark.unit
def test_coordinates_datamodule_test_full_dataset_and_scalers(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files

    dmod = CoordinatesDataModule(
        args={
            "datareader_type": "COMPRESSED_TRAJECTORY",
            "datareader_args": {
                "topology_file": pdb_path,
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
                "parallel": False,
            },
            "dataset_type": "DISTANCES",
            "train_size": 10,
            "batch_size": 5,
            "validation_size": 4,
            "test_full_dataset": True,
            "max_frames": 16,
            "norm_type": "standard",
            "num_workers": 0,
        },
        **metargs,
    )
    assert dmod.test_size == 16
    assert len(dmod.get_test_dataset()) == 16

    # Standard scaler round-trip
    raw_norm = dmod.get_train_dataset().get_norm_data()
    transformed = dmod.target_scaler_transform(raw_norm)
    recovered = dmod.target_inverse_scaler(transformed)
    np.testing.assert_allclose(raw_norm, recovered, rtol=1e-5, atol=1e-5)
    assert dmod.get_scaler_mean().shape == (15,)
    assert dmod.get_scaler_var().shape == (15,)
    assert dmod.get_scaler_scale().shape == (15,)

    # MinMax scaler round-trip
    dmod.target_scaler = None
    dmod.norm_type = "minmax"
    transformed_mm = dmod.target_scaler_transform(raw_norm)
    recovered_mm = dmod.target_inverse_scaler(transformed_mm)
    np.testing.assert_allclose(raw_norm, recovered_mm, rtol=1e-5, atol=1e-5)
    assert dmod.get_scaler_mean().shape == (15,)

    # Invalid norm_type raises ValueError
    dmod.target_scaler = None
    dmod.norm_type = "invalid_norm"
    with pytest.raises(ValueError, match="not supported"):
        dmod.fit_target_scaler()


@pytest.mark.unit
def test_coordinates_datamodule_validation_and_compatibility_errors(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files
    base_reader_args = {
        "topology_file": pdb_path,
        "trajectory_file": xtc_path,
        "topology_format": "PDB",
        "parallel": False,
    }

    # Incompatible datareader
    with pytest.raises(ValueError, match="not compatible"):
        CoordinatesDataModule(
            args={
                "datareader_type": "PLUMED_OUTPUT",
                "dataset_type": "DISTANCES",
                "train_size": 4,
                "batch_size": 2,
            },
            **metargs,
        )

    # Incompatible dataset
    with pytest.raises(ValueError, match="not compatible"):
        CoordinatesDataModule(
            args={
                "datareader_type": "COMPRESSED_TRAJECTORY",
                "datareader_args": base_reader_args,
                "dataset_type": "COLVAR",
                "train_size": 4,
                "batch_size": 2,
            },
            **metargs,
        )

    # Mutually exclusive test_size > 0 and test_full_dataset=True
    with pytest.raises(ValueError, match="mutually exclusive"):
        CoordinatesDataModule(
            args={
                "datareader_type": "COMPRESSED_TRAJECTORY",
                "datareader_args": base_reader_args,
                "dataset_type": "DISTANCES",
                "train_size": 4,
                "batch_size": 2,
                "test_size": 4,
                "test_full_dataset": True,
            },
            **metargs,
        )

    # Insufficient frames
    with pytest.raises(ValueError, match="Not enough frames"):
        CoordinatesDataModule(
            args={
                "datareader_type": "COMPRESSED_TRAJECTORY",
                "datareader_args": base_reader_args,
                "dataset_type": "DISTANCES",
                "train_size": 15,
                "validation_size": 10,
                "batch_size": 5,
            },
            **metargs,
        )

    # Batch size > train_size
    with pytest.raises(ValueError, match="Batch size must be less than or equal"):
        CoordinatesDataModule(
            args={
                "datareader_type": "COMPRESSED_TRAJECTORY",
                "datareader_args": base_reader_args,
                "dataset_type": "DISTANCES",
                "train_size": 4,
                "batch_size": 8,
            },
            **metargs,
        )


@pytest.mark.unit
def test_colvars_datamodule_compatibility_and_method_guards(plumed_colvar_file, metargs):
    # Incompatible datareader raises ValueError
    with pytest.raises(ValueError, match="not compatible"):
        ColvarsDataModule(
            args={
                "datareader_type": "COMPRESSED_TRAJECTORY",
                "dataset_type": "COLVAR",
                "train_size": 4,
                "batch_size": 2,
            },
            **metargs,
        )

    # Incompatible dataset raises ValueError
    with pytest.raises(ValueError, match="not compatible"):
        ColvarsDataModule(
            args={
                "datareader_type": "PLUMED_OUTPUT",
                "datareader_args": {"plumed_file": plumed_colvar_file},
                "dataset_type": "DISTANCES",
                "train_size": 4,
                "batch_size": 2,
            },
            **metargs,
        )

    # Coordinate-specific methods raise expected AttributeError / NotImplementedError
    dmod_stub = ColvarsDataModule.__new__(ColvarsDataModule)
    with pytest.raises(AttributeError):
        dmod_stub.get_atns()
    with pytest.raises(AttributeError):
        dmod_stub.get_bond_indices()
    with pytest.raises(AttributeError):
        dmod_stub.get_element_symbols()
    with pytest.raises(NotImplementedError):
        dmod_stub.get_fake_systems()


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bug in collective_encoder/datasets/colvar.py: "
        "ColvarDataset does not implement abstract method BaseDataset.get_data() and "
        "passes dataset_args=... to BaseDataset.__init__"
    ),
)
def test_colvars_datamodule_end_to_end(plumed_colvar_file, metargs):
    dmod = ColvarsDataModule(
        args={
            "datareader_type": "PLUMED_OUTPUT",
            "datareader_args": {
                "plumed_file": plumed_colvar_file,
                "columns": ["phi", "psi", "ene"],
            },
            "dataset_type": "COLVAR",
            "labeler_type": "COLUMN_SELECTOR",
            "labeler_args": {"columns": ["ene"]},
            "train_size": 12,
            "batch_size": 4,
            "validation_size": 6,
            "val_batch_size": 3,
            "test_size": 6,
            "test_batch_size": 3,
            "sequential": True,
            "num_workers": 0,
        },
        **metargs,
    )
    assert dmod.get_datapoint_shape() == (2,)
    assert dmod.get_label_names() == ["ene"]


@pytest.mark.unit
def test_datamodule_persistence_and_runtime_overrides(trajectory_files, tmp_path, metargs):
    pdb_path, xtc_path = trajectory_files
    config = {
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
            "train_size": 12,
            "batch_size": 4,
            "validation_size": 6,
            "val_batch_size": 3,
            "num_workers": 0,
        },
    }
    dmod = crb.load_datamodule(config, metargs)
    dmod_dir = tmp_path / "saved_dmod"
    dmod_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dmod, str(dmod_dir / "datamodule.pth"))

    # Reload with batch size & num_workers overrides
    reload_cfg = {
        "load_datamodule": str(dmod_dir),
        "datamodule_args": {
            "batch_size": 6,
            "val_batch_size": 2,
            "num_workers": 0,
        },
    }
    reloaded = crb.load_datamodule(reload_cfg, metargs)
    assert reloaded.batch_size == 6
    assert reloaded.val_batch_size == 2
    assert reloaded.num_workers == 0


# ---------------------------------------------------------------------------
# Regression / Bug Documentation Tests in Untouched Files
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_coordinates_datamodule_with_positions_dataset(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files
    dmod = CoordinatesDataModule(
        args={
            "datareader_type": "COMPRESSED_TRAJECTORY",
            "datareader_args": {
                "topology_file": pdb_path,
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
                "parallel": False,
            },
            "dataset_type": "POSITIONS",
            "train_size": 6,
            "batch_size": 2,
            "num_workers": 0,
        },
        **metargs,
    )
    assert dmod.get_datapoint_shape() == (6, 3)


@pytest.mark.unit
def test_coordinates_datamodule_explicit_dummy_labeler(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files
    dmod = CoordinatesDataModule(
        args={
            "datareader_type": "COMPRESSED_TRAJECTORY",
            "datareader_args": {
                "topology_file": pdb_path,
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
                "parallel": False,
            },
            "dataset_type": "DISTANCES",
            "labeler_type": "DUMMY",
            "train_size": 6,
            "batch_size": 2,
            "num_workers": 0,
        },
        **metargs,
    )
    assert dmod.get_datapoint_shape() == (15,)


@pytest.mark.unit
def test_coordinates_datamodule_sequential_indices_with_predict_size(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files
    dmod = CoordinatesDataModule(
        args={
            "datareader_type": "COMPRESSED_TRAJECTORY",
            "datareader_args": {
                "topology_file": pdb_path,
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
                "parallel": False,
            },
            "dataset_type": "DISTANCES",
            "train_size": 6,
            "batch_size": 2,
            "validation_size": 4,
            "test_size": 4,
            "predict_size": 2,
            "sequential": True,
            "num_workers": 0,
        },
        **metargs,
    )
    assert len(dmod.test_indices) == 4
    assert set(dmod.test_indices).isdisjoint(set(dmod.predict_indices))

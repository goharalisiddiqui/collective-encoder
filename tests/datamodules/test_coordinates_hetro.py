import os
import pytest
import numpy as np
import MDAnalysis as mda
import torch

from collective_encoder.common.registry import get_module
from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.datamodules.coordinates_hetro import (
    HeterogeneousCoordinatesDataModule,
    HetrogenousCoordinatesDataModule,
)
from collective_encoder.models.neural_nets.bge import BondGraphEncoderDecoder


@pytest.fixture
def two_species_trajectories(tmp_path, synthetic_universe, synthetic_peptide_universe):
    """
    Create two distinct trajectories (species 0: 6 atoms, 20 frames; species 1: 9 atoms, 15 frames)
    and return their PDB and XTC paths.
    """
    dir0 = tmp_path / "sp0"
    dir0.mkdir(parents=True, exist_ok=True)
    pdb0 = str(dir0 / "topol0.pdb")
    xtc0 = str(dir0 / "traj0.xtc")

    synthetic_universe.trajectory[0]
    synthetic_universe.atoms.write(pdb0)
    with mda.Writer(xtc0, synthetic_universe.atoms.n_atoms) as writer:
        for _ in synthetic_universe.trajectory:
            writer.write(synthetic_universe.atoms)

    dir1 = tmp_path / "sp1"
    dir1.mkdir(parents=True, exist_ok=True)
    pdb1 = str(dir1 / "topol1.pdb")
    xtc1 = str(dir1 / "traj1.xtc")

    synthetic_peptide_universe.trajectory[0]
    synthetic_peptide_universe.atoms.write(pdb1)
    with mda.Writer(xtc1, synthetic_peptide_universe.atoms.n_atoms) as writer:
        for _ in synthetic_peptide_universe.trajectory:
            writer.write(synthetic_peptide_universe.atoms)

    return (pdb0, xtc0), (pdb1, xtc1)


@pytest.mark.unit
def test_heterogeneous_coordinates_datamodule_basic(two_species_trajectories, metargs):
    (pdb0, xtc0), (pdb1, xtc1) = two_species_trajectories

    assert issubclass(HeterogeneousCoordinatesDataModule, CoordinatesDataModule)
    assert HetrogenousCoordinatesDataModule is HeterogeneousCoordinatesDataModule
    for key in ("HETEROGENEOUS_COORDINATES", "HETROGENOUS_COORDINATES", "HETRO_COORDINATES"):
        assert get_module("datamodules", key) is HeterogeneousCoordinatesDataModule

    dmod = HeterogeneousCoordinatesDataModule(
        args={
            "datareaders": [
                {
                    "name": "mol6",
                    "species_id": 0,
                    "datareader_type": "COMPRESSED_TRAJECTORY",
                    "datareader_args": {
                        "topology_file": pdb0,
                        "trajectory_file": xtc0,
                        "topology_format": "PDB",
                        "parallel": False,
                    },
                },
                {
                    "name": "mol9",
                    "species_id": 1,
                    "datareader_type": "COMPRESSED_TRAJECTORY",
                    "datareader_args": {
                        "topology_file": pdb1,
                        "trajectory_file": xtc1,
                        "topology_format": "PDB",
                        "parallel": False,
                    },
                },
            ],
            "dataset_type": "BONDGRAPH",
            "dataset_args": {
                "precompute_graphs": False,
            },
            "train_size": 12,
            "batch_size": 4,
            "validation_size": 4,
            "val_batch_size": 2,
            "test_size": 4,
            "test_batch_size": 2,
            "predict_size": 2,
            "predict_batch_size": 2,
            "sequential": False,
            "mixing_strategy": "balanced",
            "num_workers": 0,
        },
        **metargs,
    )

    assert dmod.get_num_species() == 2
    assert dmod.get_species_names() == ["mol6", "mol9"]
    assert dmod.get_species_ids() == [0, 1]

    atns = dmod.get_atns()
    assert len(atns) == 2
    assert len(atns[0]) == 6
    assert len(atns[1]) == 9
    assert len(dmod.get_atns(0)) == 6
    assert len(dmod.get_atns("mol9")) == 9

    bonds = dmod.get_bond_indices()
    assert len(bonds) == 2
    assert len(bonds[0]) == 5
    assert len(bonds[1]) == 8
    assert len(dmod.get_bond_indices("mol6")) == 5

    assert dmod.get_label_names() == ["species_id"]

    loader = dmod.train_dataloader()
    batch = next(iter(loader))
    assert hasattr(batch, "species_id")
    assert batch.species_id.numel() == 4
    assert batch.y.shape == (4, 1)
    for sid in batch.species_id:
        assert sid.item() in (0, 1)


@pytest.mark.unit
def test_heterogeneous_coordinates_labeler_override_and_graph_analyser(
    two_species_trajectories, metargs
):
    (pdb0, xtc0), (pdb1, xtc1) = two_species_trajectories

    dmod = HeterogeneousCoordinatesDataModule(
        args={
            "datareaders": [
                {
                    "name": "mol6_unlabeled",
                    "datareader_type": "COMPRESSED_TRAJECTORY",
                    "datareader_args": {
                        "topology_file": pdb0,
                        "trajectory_file": xtc0,
                        "topology_format": "PDB",
                        "parallel": False,
                    },
                    "labeler_type": None,
                },
                {
                    "name": "mol9_labeled",
                    "datareader_type": "COMPRESSED_TRAJECTORY",
                    "datareader_args": {
                        "topology_file": pdb1,
                        "trajectory_file": xtc1,
                        "topology_format": "PDB",
                        "parallel": False,
                    },
                },
            ],
            "labeler_type": "DIHEDRAL",
            "labeler_args": {
                "label_dihedrals": [
                    {
                        "phi_2": "(resid 1 and name C) or (resid 2 and (name N or name CA or name C))"
                    },
                    {
                        "psi_2": "(resid 2 and (name N or name CA or name C)) or (resid 3 and name N)"
                    },
                ],
            },
            "dataset_type": "BONDGRAPH",
            "dataset_args": {
                "precompute_graphs": False,
            },
            "train_size": 8,
            "batch_size": 4,
            "validation_size": 2,
            "val_batch_size": 2,
            "test_size": 2,
            "test_batch_size": 2,
            "sequential": False,
            "num_workers": 0,
            "data_analysers": [
                {
                    "analyser_type": "GRAPH",
                    "analyser_args": {
                        "labels_selection": {
                            "phi": [
                                "y",
                                0,
                            ],
                            "psi": [
                                "y",
                                1,
                            ],
                        },
                        "plots_2dscatter_cb": [
                            {
                                "name": "rama_test",
                                "x": "phi",
                                "y": "psi",
                            }
                        ],
                    },
                }
            ],
        },
        **metargs,
    )

    label_names = dmod.get_label_names()
    assert label_names == ["phi_2", "psi_2", "species_id"]

    # Check 2-D PyG batch collation (B, 3) when n_labels == 3
    batch = next(iter(dmod.train_dataloader()))
    assert batch.y.shape == (4, 3)


@pytest.mark.unit
def test_multispecies_bge_forward_loss_metrics_and_decode(
    two_species_trajectories, metargs
):
    (pdb0, xtc0), (pdb1, xtc1) = two_species_trajectories

    dmod = HeterogeneousCoordinatesDataModule(
        args={
            "datareaders": [
                {
                    "name": "mol6",
                    "species_id": 0,
                    "datareader_type": "COMPRESSED_TRAJECTORY",
                    "datareader_args": {
                        "topology_file": pdb0,
                        "trajectory_file": xtc0,
                        "topology_format": "PDB",
                        "parallel": False,
                    },
                },
                {
                    "name": "mol9",
                    "species_id": 1,
                    "datareader_type": "COMPRESSED_TRAJECTORY",
                    "datareader_args": {
                        "topology_file": pdb1,
                        "trajectory_file": xtc1,
                        "topology_format": "PDB",
                        "parallel": False,
                    },
                },
            ],
            "dataset_type": "BONDGRAPH",
            "dataset_args": {
                "precompute_graphs": False,
            },
            "train_size": 8,
            "batch_size": 4,
            "validation_size": 4,
            "val_batch_size": 4,
            "test_size": 4,
            "test_batch_size": 4,
            "sequential": False,
            "num_workers": 0,
        },
        **metargs,
    )

    raw_bge_args = {
        "loss_weights": [0.05, 0.05, 0.45, 0.45],
        "encoder_args": {
            "node_embed_dim": 4,
            "edge_embed_dim": 4,
            "hidden_dim": 16,
            "num_layers": 2,
            "heads": 2,
            "set2set_steps": 1,
            "latent_dim": 8,
            "dropout": 0.0,
        },
        "decoder_args": {
            "template_khop": 2,
            "node_embed_dim": 4,
            "edge_embed_dim": 4,
            "hidden_dim": 16,
            "num_layers": 1,
            "heads": 1,
            "dropout_mp": 0.0,
            "dropout_mlp": 0.0,
            "final_mlp_layers": 1,
            "rbf_dim": 8,
            "rbf_min": 0.0,
            "rbf_max": 4.0,
            "rbf_gamma": 10.0,
        },
    }
    bge_args = BondGraphEncoderDecoder.extract_args_from_datamodule(dmod, raw_bge_args)
    model = BondGraphEncoderDecoder(args=bge_args, **metargs)
    assert model.decoder_net.is_multispecies is True

    batch = next(iter(dmod.train_dataloader()))
    output, latent, meta = model(batch)
    assert latent.shape == (4, 8)
    assert "species_id" in meta
    assert output["bond_dist"].shape == (4, model.decoder_net.max_bonds)

    # Verify NaN-masked loss and metrics on mixed-species batch
    labels = model.extract_labels(batch)
    loss, loss_dict = model.losses["encdec"](batch, latent, output, labels, meta)
    assert torch.isfinite(loss)
    loss.backward()

    mae_val, mae_dict = model.metrics["mae"].calculate(batch, latent, output, labels, meta)
    assert torch.isfinite(mae_val)

    # Verify get_decoded with and without explicit species_id
    dec_with_sid = model.get_decoded(latent, species_id=batch.species_id)
    assert dec_with_sid["bond_dist"].shape == (4, model.decoder_net.max_bonds)
    for i, sid in enumerate(batch.species_id.view(-1)):
        n_b = len(model.decoder_net.species_bond_index[int(sid.item())])
        assert torch.all(torch.isfinite(dec_with_sid["bond_dist"][i, :n_b]))
        if n_b < model.decoder_net.max_bonds:
            assert torch.all(torch.isnan(dec_with_sid["bond_dist"][i, n_b:]))

    dec_default = model.get_decoded(latent)
    assert dec_default["bond_dist"].shape == (4, model.decoder_net.max_bonds)

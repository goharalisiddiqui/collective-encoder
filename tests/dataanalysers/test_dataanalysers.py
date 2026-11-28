import os
import pytest

from collective_encoder.dataanalysers.simple import SimpleDataAnalyser
from collective_encoder.dataanalysers.graph import GraphDataAnalyser
from collective_encoder.datasets.distances import DistancesDataset
from collective_encoder.datasets.bondgraph import BondGraphDataset


@pytest.mark.unit
def test_simple_data_analyser_parse_and_2dscatter(synthetic_atoms_list, metargs):
    structures, labels = synthetic_atoms_list
    ds = DistancesDataset(
        structures=structures[:8],
        labels=labels[:8],
        args={"parallel": False},
        **metargs,
    )

    analyser = SimpleDataAnalyser(
        args={
            "datamodule_args": {"dataset_type": "DISTANCES"},
            "datamodule_labels_list": ["sin_t", "cos_t"],
            "data_selection": {"d0": 0, "d1": 1},
            "labels_selection": {"s": 0, "c": 1},
            "plots_2dscatter_cb": [
                {"x": "d0", "y": "d1", "color": "s", "name": "dist_scatter"},
            ],
        },
        **metargs,
    )
    analyser.write_data(ds, label="train")
    expected_png = os.path.join(analyser.outpath, "dist_scatter_train.png")
    assert os.path.isfile(expected_png)

    # Incompatible dataset_type raises ValueError
    with pytest.raises(ValueError, match="not compatible"):
        SimpleDataAnalyser(
            args={
                "datamodule_args": {"dataset_type": "BONDGRAPH"},
                "datamodule_labels_list": [],
            },
            **metargs,
        )


@pytest.mark.unit
def test_graph_data_analyser_parse_data(synthetic_atoms_list, metargs):
    import numpy as np

    structures, _ = synthetic_atoms_list
    dict_labels = {
        "bonds": np.ones((6, 3), dtype=float),
        "angles": np.ones((6, 4), dtype=float),
    }
    ds = BondGraphDataset(
        structures=structures[:6],
        labels=dict_labels,
        args={
            "bond_indices": [(0, 1), (1, 2), (2, 3)],
            "precompute_graphs": False,
            "parallel": False,
        },
        **metargs,
    )

    analyser = GraphDataAnalyser(
        args={
            "datamodule_args": {"dataset_type": "BONDGRAPH"},
            "datamodule_labels_list": ["bonds", "angles"],
            "parallel": True,
            "labels_selection": {"b0": ["y_bonds", 0], "b1": ["y_bonds", 1]},
            "plots_2dscatter_cb": [
                {"x": "b0", "y": "b1", "name": "bond_scatter"},
            ],
            "plots_hist_cb": [
                {"x": "b0", "name": "hist_bonds"},
                {"x": "b1", "name": "hist_b1"},
            ],
        },
        **metargs,
    )
    data_dict, labels_dict = analyser.parse_data(ds)
    assert "x" in data_dict
    assert "y_bonds" in labels_dict
    assert labels_dict["y_bonds"].shape == (6, 3)

    analyser.write_data(ds, label="train")
    assert os.path.isfile(os.path.join(analyser.outpath, "bond_scatter_train.png"))
    assert os.path.isfile(os.path.join(analyser.outpath, "hist_bonds_train.png"))
    assert os.path.isfile(os.path.join(analyser.outpath, "hist_b1_train.png"))


@pytest.mark.unit
def test_simple_plotter_topology_parity_grouped_by_species(metargs):
    import numpy as np
    from collective_encoder.testplotters.simple import SimplePlotter

    n = 12
    true_bonds = np.random.uniform(1.2, 1.6, size=(n, 5))
    true_bonds[:4, 3:] = np.nan
    pred_bonds = true_bonds + np.random.normal(0.0, 0.02, size=(n, 5))

    true_angles = np.random.uniform(1.8, 2.2, size=(n, 6))
    pred_angles = true_angles + np.random.normal(0.0, 0.05, size=(n, 6))

    true_dih = np.random.uniform(-np.pi, np.pi, size=(n, 4))
    pred_dih = true_dih + np.random.normal(0.0, 0.1, size=(n, 4))

    species_id = np.array([0] * 4 + [1] * 4 + [2] * 4)[:, None]

    labels = {
        "bond_dist": true_bonds,
        "angle": true_angles,
        "dihedral_cos": np.cos(true_dih),
        "dihedral_sin": np.sin(true_dih),
        "species_id": species_id,
    }
    pred = {
        "bond_dist": pred_bonds,
        "angle": pred_angles,
        "dihedral_cos": np.cos(pred_dih),
        "dihedral_sin": np.sin(pred_dih),
    }

    plotter = SimplePlotter(
        args={
            "labels_selection": {
                "true_bonds": "bond_dist",
                "true_angles": "angle",
                "true_dihedrals_cos": "dihedral_cos",
                "true_dihedrals_sin": "dihedral_sin",
                "species_id": "species_id",
            },
            "pred_selection": {
                "pred_bonds": "bond_dist",
                "pred_angles": "angle",
                "pred_dihedrals_cos": "dihedral_cos",
                "pred_dihedrals_sin": "dihedral_sin",
            },
            "plots_2dscatter_cb": [
                {
                    "name": "true_vs_pred_bonds",
                    "x": "true_bonds",
                    "y": "pred_bonds",
                    "diagonal": True,
                    "group_by": "species_id",
                    "group_names": {0: "2Ala", 1: "3Ala", 2: "4Ala"},
                },
                {
                    "name": "true_vs_pred_dihedrals",
                    "x": "true_dihedrals",
                    "y": "pred_dihedrals",
                    "angles": True,
                    "diagonal": True,
                    "group_by": "species_id",
                    "group_mode": "files",
                    "group_names": {0: "2Ala", 1: "3Ala", 2: "4Ala"},
                },
            ],
        },
        **metargs,
    )
    plotter.add_batch(
        data=None,
        latent=np.zeros((n, 2)),
        pred=pred,
        labels=labels,
        meta={},
    )
    plotter.finish()

    assert os.path.isfile(os.path.join(plotter.outpath, "true_vs_pred_bonds.png"))
    assert os.path.isfile(os.path.join(plotter.outpath, "true_vs_pred_dihedrals_2Ala.png"))
    assert os.path.isfile(os.path.join(plotter.outpath, "true_vs_pred_dihedrals_3Ala.png"))
    assert os.path.isfile(os.path.join(plotter.outpath, "true_vs_pred_dihedrals_4Ala.png"))

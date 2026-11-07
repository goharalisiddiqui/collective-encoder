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
    structures, labels = synthetic_atoms_list
    ds = BondGraphDataset(
        structures=structures[:6],
        labels=labels[:6],
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
            "datamodule_labels_list": ["sin_t", "cos_t"],
            "labels_selection": {"b0": ["y_bonds", 0], "b1": ["y_bonds", 1]},
            "plots_2dscatter_cb": [
                {"x": "b0", "y": "b1", "name": "bond_scatter"},
            ],
        },
        **metargs,
    )
    graphs = ds.get_data()
    data_dict, labels_dict = analyser.parse_data(graphs)
    assert "x" in data_dict
    assert "y_bonds" in labels_dict
    assert labels_dict["y_bonds"].shape == (6, 3)

    analyser.write_data(graphs, label="train")
    assert os.path.isfile(os.path.join(analyser.outpath, "bond_scatter_train.png"))

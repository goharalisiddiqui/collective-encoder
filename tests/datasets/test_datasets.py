import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from collective_encoder.datasets.positions import PositionsDataset
from collective_encoder.datasets.distances import DistancesDataset
from collective_encoder.datasets.bondgraph import BondGraphDataset
from collective_encoder.datasets.colvar import ColvarDataset
from collective_encoder.datasets.resolver import get_dataset_cls_dl


@pytest.mark.unit
def test_positions_dataset(synthetic_atoms_list, metargs):
    structures, labels = synthetic_atoms_list

    ds_raw = PositionsDataset(
        structures=structures[:6],
        labels=labels[:6],
        args={},
        **metargs,
    )
    assert len(ds_raw) == 6
    assert ds_raw.get_datapoint_shape() == (4, 3)
    x0, y0 = ds_raw[0]
    assert x0.shape == (4, 3)
    assert y0.shape == (2,)
    assert ds_raw.get_norm_data().shape == (6, 4, 3)
    pos_arr, lbl_arr = ds_raw.get_data()
    assert pos_arr.shape == (6, 4, 3)
    assert lbl_arr.shape == (6, 2)


@pytest.mark.unit
def test_distances_dataset(synthetic_atoms_list, metargs):
    structures, labels = synthetic_atoms_list

    # Default upper-triangular pairwise distances: 4 * 3 // 2 = 6
    ds = DistancesDataset(
        structures=structures[:6],
        labels=labels[:6],
        args={},
        **metargs,
    )
    assert len(ds) == 6
    assert ds.get_datapoint_shape() == (6,)
    x0, y0 = ds[0]
    assert x0.shape == (6,)
    assert torch.all(x0 > 0)
    assert y0.shape == (2,)
    assert ds.get_norm_data().shape == (6, 6)
    d_arr, l_arr = ds.get_data()
    assert d_arr.shape == (6, 6)
    assert l_arr.shape == (6, 2)

    # Subset via slice notation group1="0:2", group2="2:4" -> 2 * 2 = 4 pairs
    ds_sub = DistancesDataset(
        structures=structures[:6],
        labels=labels[:6],
        args={"group1": "0:2", "group2": "2:4", "atm_ids": [1, 2, 3, 4]},
        **metargs,
    )
    assert ds_sub.get_datapoint_shape() == (4,)


@pytest.mark.unit
def test_bondgraph_dataset_on_the_fly_and_precomputed(synthetic_atoms_list, metargs):
    structures, labels = synthetic_atoms_list
    bonds = [(0, 1), (1, 2), (2, 3)]

    # 1. On-the-fly graph construction (precompute_graphs=False)
    ds_fly = BondGraphDataset(
        structures=structures[:5],
        labels=labels[:5],
        args={"bond_indices": bonds, "precompute_graphs": False, "parallel": False},
        tag="train_fly",
        **metargs,
    )
    assert len(ds_fly) == 5
    assert ds_fly.get_num_node_features() == 3
    assert ds_fly.get_num_edge_features() == 3
    g0 = ds_fly[0]
    assert g0.x.shape == (3, 3)  # 3 bonds, 3 node features
    assert g0.edge_index.shape[0] == 2
    assert g0.y_bonds.shape == (3,)
    assert g0.y_angles.shape == (4,)  # 2 angles * 2 directed edges
    assert g0.y_torsions_sin.shape == (2,)  # 1 torsion * 2 directed edges
    assert g0.y_torsions_cos.shape == (2,)

    # Template graph and label indices
    bond_idx, angle_idx, torsion_idx = ds_fly.get_label_indices()
    assert len(bond_idx) == 3
    assert len(angle_idx) == 4
    assert len(torsion_idx) == 2
    tmpl = ds_fly.get_template_graph(k=2)
    assert tmpl.x.shape[0] == 4  # 4 atoms in template graph
    assert ds_fly.get_norm_data().shape[0] == 5

    # 2. Precomputed graph mode (precompute_graphs=True, parallel=False)
    ds_pre = BondGraphDataset(
        structures=structures[:5],
        labels=labels[:5],
        args={"bond_indices": bonds, "precompute_graphs": True, "parallel": False},
        tag="train_pre",
        **metargs,
    )
    g0_pre = ds_pre[0]
    torch.testing.assert_close(g0.x, g0_pre.x)
    torch.testing.assert_close(g0.edge_attr, g0_pre.edge_attr)

    # Empty bond_indices raises ValueError
    with pytest.raises(ValueError, match="bond_indices cannot be empty"):
        BondGraphDataset(
            structures=structures[:2],
            labels=labels[:2],
            args={"bond_indices": [], "precompute_graphs": False},
            **metargs,
        )


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bug in collective_encoder/datasets/colvar.py: "
        "ColvarDataset does not implement abstract method BaseDataset.get_data() and "
        "passes dataset_args=... to BaseDataset.__init__"
    ),
)
def test_colvar_dataset(metargs):
    rng = np.random.default_rng(11)
    data = rng.normal(size=(10, 3)).astype(np.float32)
    labels = rng.normal(size=(10, 2)).astype(np.float32)

    ds = ColvarDataset(
        structures=data,
        labels=labels,
        args={},
        **metargs,
    )
    assert len(ds) == 10
    assert ds.get_datapoint_shape() == (3,)
    x0, y0 = ds[0]
    assert x0.shape == (3,)
    assert y0.shape == (2,)
    assert ds.get_norm_data().shape == (10, 3)
    d_arr, l_arr = ds.get_data()
    assert d_arr.shape == (10, 3)
    assert l_arr.shape == (10, 2)


@pytest.mark.unit
def test_dataset_resolver(trajectory_files, plumed_colvar_file, metargs):
    from collective_encoder.datareaders.compressed_trajectory import CompressedTrajectoryReader
    from collective_encoder.datareaders.plumed_output import PlumedOutputReader

    pdb_path, xtc_path = trajectory_files
    coord_reader = CompressedTrajectoryReader(
        args={
            "topology_file": pdb_path,
            "trajectory_file": xtc_path,
            "topology_format": "PDB",
            "parallel": False,
        },
        **metargs,
    )

    cls_pos, args_pos, dl_pos = get_dataset_cls_dl("POSITIONS", {}, coord_reader)
    assert cls_pos is PositionsDataset
    assert dl_pos is DataLoader

    cls_bg, args_bg, dl_bg = get_dataset_cls_dl("BONDGRAPH", {}, coord_reader)
    assert cls_bg is BondGraphDataset
    assert dl_bg is GeoDataLoader
    assert "bond_indices" in args_bg

    plumed_reader = PlumedOutputReader(
        args={"plumed_file": plumed_colvar_file, "columns": ["phi", "psi"]},
        **metargs,
    )
    cls_cv, args_cv, dl_cv = get_dataset_cls_dl("COLVAR", {}, plumed_reader)
    assert cls_cv is ColvarDataset
    assert dl_cv is DataLoader

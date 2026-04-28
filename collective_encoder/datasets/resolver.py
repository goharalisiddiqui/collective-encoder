import importlib
import logging
from typing import Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry maps a dataset identifier to a factory function that returns
#   (dataset_class, modified_dataset_args, dataloader_class)
# Using factories rather than plain classes lets each type add its own
# datareader-specific args (e.g. bond indices for GRAPH datasets).
# ---------------------------------------------------------------------------

def _make_positions(dataset_args: Dict, datareader):
    from collective_encoder.datasets.positions import PositionsDataset
    return PositionsDataset, dataset_args, DataLoader


def _make_distances(dataset_args: Dict, datareader):
    from collective_encoder.datasets.distances import DistancesDataset
    args = {**dataset_args, "atm_ids": datareader.atm_ids}
    return DistancesDataset, args, DataLoader


def _make_graph(dataset_args: Dict, datareader):
    from collective_encoder.datasets.bondgraph import BondGraphDataset
    args = {**dataset_args, "bond_indices": datareader.bonds}
    return BondGraphDataset, args, GeoDataLoader


def _make_graph_latent(dataset_args: Dict, datareader):
    from collective_encoder.datasets.bondgraph_latent import BondGraphLatentDataset
    args = {**dataset_args, "bond_indices": datareader.bonds}
    return BondGraphLatentDataset, args, DataLoader


def _make_soap(dataset_args: Dict, datareader):
    from collective_encoder.datasets.soap import SOAPDataset
    args = _soap_modifications({**dataset_args}, datareader)
    return SOAPDataset, args, DataLoader


def _make_soap_ps(dataset_args: Dict, datareader):
    from collective_encoder.datasets.soap_ps import SoapPowerSpectrumDataset
    args = _soap_modifications({**dataset_args}, datareader)
    return SoapPowerSpectrumDataset, args, DataLoader


def _make_colvar(dataset_args: Dict, datareader):
    from collective_encoder.datasets.colvar import ColvarDataset
    return ColvarDataset, dataset_args, DataLoader


_REGISTRY = {
    "DEFAULT":    _make_positions,
    "POSITIONS":  _make_positions,
    "DISTANCES":  _make_distances,
    "GRAPH":      _make_graph,
    "GRAPH_LATENT": _make_graph_latent,
    "SOAP":       _make_soap,
    "SOAP_PS":    _make_soap_ps,
    "COLVAR":     _make_colvar,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_cls_dl(dataset_type: str, dataset_args: Dict, datareader):
    """Return ``(dataset_class, dataset_args, dataloader_class)`` for *dataset_type*.

    Raises:
        ValueError: If *dataset_type* is not registered.
    """
    if dataset_type not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[dataset_type](dataset_args, datareader)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _soap_modifications(dataset_args: Dict, datareader) -> Dict:
    """Resolve MDAnalysis atom selections to integer indices for SOAP datasets."""
    if dataset_args.get("atoms_selections") is None:
        return dataset_args

    u = datareader.univ
    mol = datareader.mol
    selected_indices = []

    for selection in dataset_args["atoms_selections"]:
        sel_atoms = u.select_atoms(selection)
        if sel_atoms.n_atoms != 1:
            _log.warning(
                "[SOAP] Selection '%s' does not select exactly one atom, selected %d atoms",
                selection, sel_atoms.n_atoms,
            )
        n_types = len({at.type for at in sel_atoms})
        if n_types > 1:
            _log.warning("[SOAP] Selection '%s' selects more than one atom type.", selection)
        _log.info("Selected %d atoms of total %d types", sel_atoms.n_atoms, n_types)
        for at in sel_atoms:
            mol_index = np.where(mol.atoms.indices == at.index)[0]
            if len(mol_index) == 0:
                raise ValueError(
                    f"Atom {at.index} in selection '{selection}' not found in "
                    "selected molecule atoms"
                )
            selected_indices.append(int(mol_index[0]))

    dataset_args["selected_atoms"] = dataset_args.get("selected_atoms", []) + selected_indices
    del dataset_args["atoms_selections"]
    return dataset_args

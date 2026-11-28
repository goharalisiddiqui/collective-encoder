import os
from multiprocessing import Pool
from typing import Any, List, Tuple, Dict, Optional, Union
import math

import numpy as np
import ase

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from collective_encoder.datasets.base_graph import BaseGraphDataset
from collective_encoder import TENSOR_PRECISION
from collective_encoder.datareaders.base import BaseDataReader

from .utils.bondgraph import (assemble_bonds, bond_node_features, 
                              angle_and_torsion_edges,
                              build_template_graph)

from gslibs.utils.filesystem import create_rundir


def _compute_graph(structure: ase.Atoms, bond_indices: List[Tuple[int, int]]) -> Data:
    """Standalone graph computation from an ASE Atoms object and bond index list.
    Separated from the class so it can be called from worker processes without pickling self.
    """
    struct_bonds = structure.info.get("bonds", bond_indices)
    struct_bonds = [tuple(map(int, b)) for b in struct_bonds]
    bonds = assemble_bonds(structure, struct_bonds)
    x = bond_node_features(structure, bonds)
    edge_index, edge_attr, _, angle_atoms, torsion_atoms = angle_and_torsion_edges(structure, bonds)
    positions = torch.tensor(structure.get_positions(), dtype=TENSOR_PRECISION)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    if "species_id" in structure.info:
        data.species_id = torch.tensor([int(structure.info["species_id"])], dtype=torch.long)
    return data


def _attach_graph_label(
    graph: Data,
    label: Any,
    dtype: torch.dtype = TENSOR_PRECISION,
    prefix: str = "y",
) -> Data:
    """
    Attach a per-structure label (dict or array-like) as native tensor attribute(s)
    on a PyG :class:`Data` object.

    When ``label`` is a dictionary, each key ``k`` is attached as ``graph[f"{prefix}_{k}"]``
    (or ``graph[k]`` if ``k`` already begins with ``f"{prefix}_"``) with shape ``(1, D_k)``.
    """
    if label is None:
        return graph
    if isinstance(label, dict):
        for k, v in label.items():
            attr_name = k if k.startswith(f"{prefix}_") else f"{prefix}_{k}"
            graph[attr_name] = torch.as_tensor(v, dtype=dtype).reshape(1, -1)
    else:
        graph[prefix] = torch.as_tensor(label, dtype=dtype)
    return graph


def _compute_graph_worker(args):
    """Worker: computes and saves graphs for a chunk of (global_idx, structure, label) items.
    Takes a single packed tuple so it is compatible with pool.imap.
    """
    worker_id, items, bond_indices, graphs_dir, verbose = args

    for global_idx, structure, label in tqdm(items,
                                             position=worker_id,
                                             desc=f"Worker {worker_id}",
                                             leave=False,
                                             disable=not verbose,
                                             dynamic_ncols=True):
        graph = _compute_graph(structure, bond_indices)
        if label is not None:
            _attach_graph_label(graph, label, TENSOR_PRECISION)
        torch.save(graph, os.path.join(graphs_dir, f"graph_{global_idx}.pt"))


class BondGraphDataset(BaseGraphDataset, Dataset):
    """
    Graph dataset where nodes are bonds and edges connect bonds via angles or torsions.

    Node (bond) feature vector (length 3):
        [Z_i, Z_j, bond_length]

    Edge feature vector (length 3):
        [is_angle_edge, is_torsion_edge, value]
        value = bond angle (radians) for angle edges, dihedral angle (radians) for torsion edges.

    Edges are bidirectional (both directions added with same attributes).

    Parameters
    ----------
    structures : list of ase.Atoms
        A list of molecular structures.
    labels : list of float, optional
        A list of target labels corresponding to each structure.
    dataset_args : dict, optional
        Configuration dictionary. Recognised keys include:
        - ``bond_indices`` (required): list of (i,j) pairs.
        - ``precompute_graphs`` (bool, default True): Precompute graphs to disk.
        - ``parallel`` (bool, default True): Precompute using multiple processes.
    kwargs : dict
        Additional keyword arguments (e.g. ``tag``).
    """

    _IDENTIFIER = "BONDGRAPH"
    _REQUIRED_ARGS = [
        "bond_indices"
    ]
    
    @classmethod
    def _prepare_dataset(cls, 
                         dataset_args: Dict[str, Union[float, int, str]], 
                         datareader: BaseDataReader) \
        -> Tuple[type, Dict[str, Union[float, int, str]], type]:
        return cls, {**dataset_args, "bond_indices": datareader.get_bonds()}, GeoDataLoader

    def __init__(
            self,
            structures: List[ase.Atoms],
            labels: Optional[Union[Dict[str, Union[torch.Tensor, np.ndarray, List]], List[float]]] = None,
            args: Dict[str, Union[float, int, str]] = None,
            **kwargs,
        ):
        BaseGraphDataset.__init__(self, args=args, **kwargs)
        Dataset.__init__(self)
        
        self.structures = structures
        self.labels = self._validate_and_prepare_labels(labels, len(structures))
        self.tag = kwargs.get("tag", None)
        
        if self.bond_indices is None or len(self.bond_indices) == 0:
            self.raise_error("bond_indices cannot be empty.")

        # Support per-structure bond indices (e.g. from multi-molecule datasets)
        if any(s.info.get("bonds") is not None for s in structures):
            for s in structures:
                if s.info.get("bonds") is not None:
                    s.info["bonds"] = [tuple(map(int, b)) for b in s.info["bonds"]]
                    n_atoms_s = len(s)
                    for (i, j) in s.info["bonds"]:
                        assert 0 <= i < n_atoms_s and 0 <= j < n_atoms_s, (
                            f"Bond index out of range: ({i},{j}) for n_atoms={n_atoms_s}"
                        )
            if structures[0].info.get("bonds") is not None:
                self.bond_indices = list(structures[0].info["bonds"])
            else:
                self.bond_indices = [tuple(map(int, b)) for b in self.bond_indices]
        else:
            if (
                isinstance(self.bond_indices[0], (list, tuple))
                and len(self.bond_indices[0]) > 0
                and isinstance(self.bond_indices[0][0], (list, tuple))
            ):
                self.bond_indices = self.bond_indices[0]
            self.bond_indices = [tuple(map(int, b)) for b in self.bond_indices]  # Ensure bond indices are tuples of ints
            # Validate bond indices
            n_atoms0 = len(structures[0])
            for (i, j) in self.bond_indices:
                assert 0 <= i < n_atoms0 and 0 <= j < n_atoms0, f"Bond index out of range: ({i},{j}) for n_atoms={n_atoms0}"

        self._calculate_label_indices()
        unique_angles = list(dict.fromkeys(tuple(a) for a in self.angle_index))
        unique_torsions = list(dict.fromkeys(tuple(t) for t in self.torsion_index))
        self.log_info(
            f"Topology internal coordinate counts (reference structure): "
            f"{len(self.bond_indices)} bonds, "
            f"{len(unique_angles)} unique angles ({len(self.angle_index)} directed), "
            f"{len(unique_torsions)} unique torsions ({len(self.torsion_index)} directed)"
        )
        self.log_debug(f"Bond set (atom index pairs): {self.bond_indices}")
        self.log_debug(f"Angle set (unique atom index triplets): {unique_angles}")
        self.log_debug(f"Torsion set (unique atom index quartets): {unique_torsions}")
    
        self._post_init()
    
    def _validate_and_prepare_labels(
        self,
        labels: Optional[Union[Dict[str, Any], List[Any], np.ndarray, torch.Tensor]],
        n_structures: int,
    ):
        """
        Validate that ``labels`` matches ``n_structures`` along the first dimension.

        When ``labels`` is a dictionary, validates that every key's tensor/array/list
        has ``shape[0] == n_structures`` and converts each value to a 2-D tensor of
        shape ``(n_structures, D_k)``.
        """
        if labels is None:
            return None

        if isinstance(labels, dict):
            validated: Dict[str, torch.Tensor] = {}
            for key, val in labels.items():
                t = torch.as_tensor(val, dtype=self.dtype)
                if t.ndim == 1:
                    t = t.unsqueeze(1)
                if t.shape[0] != n_structures:
                    self.raise_error(
                        f"Label '{key}' first dimension ({t.shape[0]}) must equal "
                        f"the number of structures ({n_structures})."
                    )
                validated[key] = t
            return validated

        if len(labels) != n_structures:
            self.raise_error(
                f"Number of labels ({len(labels)}) must equal "
                f"the number of structures ({n_structures})."
            )
        return labels

    def _get_label_at(self, idx: int):
        """Extract the label (dict or array/tensor) for structure ``idx``."""
        if self.labels is None:
            return None
        if isinstance(self.labels, dict):
            return {k: v[idx].reshape(1, -1) for k, v in self.labels.items()}
        return self.labels[idx]

    def _compute_graph(self, idx: int) -> Data:
        graph = _compute_graph(self.structures[idx], self.bond_indices)
        lbl = self._get_label_at(idx)
        if lbl is not None:
            _attach_graph_label(graph, lbl, self.dtype, prefix=self._GRAPH_LABEL_PREFIX)
        return graph

    def _calculate_label_indices(self):
        atoms = self.structures[0]
        bonds = assemble_bonds(atoms, self.bond_indices)
        edge_index, edge_attr, _, angle_atoms, torsion_atoms = angle_and_torsion_edges(atoms, bonds)
        # Store bond <-> atom mapping
        self.bond_index = [[i, j] for (i, j, _) in bonds]
        self.angle_index = angle_atoms
        self.torsion_index = torsion_atoms

        # Track per-species representative structure, bonds, and label indices
        self.species_structures: Dict[int, ase.Atoms] = {}
        self.species_bond_indices: Dict[int, List[Tuple[int, int]]] = {}
        self.species_label_indices: Dict[int, tuple] = {}
        for s in self.structures:
            sid = s.info.get("species_id")
            if sid is not None:
                sid = int(sid)
                if sid not in self.species_structures:
                    s_bonds = [tuple(map(int, b)) for b in s.info.get("bonds", self.bond_indices)]
                    assembled = assemble_bonds(s, s_bonds)
                    _, _, _, s_angles, s_torsions = angle_and_torsion_edges(s, assembled)
                    self.species_structures[sid] = s
                    self.species_bond_indices[sid] = s_bonds
                    self.species_label_indices[sid] = (
                        [[i, j] for (i, j, _) in assembled],
                        s_angles,
                        s_torsions,
                    )
    
    def _precompute_graphs(self):
        precomputed_graphs_dir = os.path.join(self.run_dir, f"precomputed_graphs_{self.tag}")
        if self.tag is not None and not os.path.exists(precomputed_graphs_dir):
            os.makedirs(precomputed_graphs_dir, exist_ok=True)
        else:
            precomputed_graphs_dir = create_rundir(self.run_dir, "precomputed_graphs", 0, overwrite=False)

        labels_list = [self._get_label_at(i) for i in range(self.len())]
        items = [(idx, struct, lbl)
                 for idx, (struct, lbl) in enumerate(zip(self.structures, labels_list))]

        if not self.parallel:
            for idx, structure, label in tqdm(items, desc="Precomputing graphs",
                                              disable=not self.verbose, dynamic_ncols=True):
                graph = _compute_graph(structure, self.bond_indices)
                if label is not None:
                    _attach_graph_label(graph, label, self.dtype, prefix=self._GRAPH_LABEL_PREFIX)
                torch.save(graph, os.path.join(precomputed_graphs_dir, f"graph_{idx}.pt"))
        else:
            n_workers = min(16, os.cpu_count() or 1, max(1, len(items)))
            chunks = [items[i::n_workers] for i in range(n_workers)]
            chunks = [c for c in chunks if c]

            args = [
                (i, chunk, self.bond_indices, precomputed_graphs_dir, self.verbose)
                for i, chunk in enumerate(chunks)
            ]

            with Pool(processes=len(args)) as pool:
                list(tqdm(
                    pool.imap(_compute_graph_worker, args),
                    total=len(args),
                    position=len(args),
                    desc="Precomputing graphs",
                    leave=True,
                    disable=not self.verbose,
                    dynamic_ncols=True,
                ))

        self.precompute_graphs_dir = precomputed_graphs_dir

    def get_label_indices(self, species_id: Optional[int] = None) -> tuple:
        """
        Return list of atom indices of labels (optionally for a specific species).

        Parameters
        ----------
        species_id : int, optional
            Species identifier. If ``None``, returns label indices for the default
            (first) structure.

        Returns
        -------
        tuple
            Tuple containing lists of bond, angle, and torsion indices.
        """
        if species_id is not None and hasattr(self, "species_label_indices") and species_id in self.species_label_indices:
            return self.species_label_indices[species_id]
        return self.bond_index, self.angle_index, self.torsion_index

    def get_species_label_indices(self) -> Dict[int, tuple]:
        """
        Return mapping from ``species_id`` to ``(bond_index, angle_index, torsion_index)``.

        Returns
        -------
        dict of int to tuple
            Per-species label index tuples.
        """
        return dict(getattr(self, "species_label_indices", {}))

    def get_template_graph(self, 
                           k: int = 3,
                           residue_names: Optional[List[str]] = None,
                           atom_names: Optional[List[str]] = None,
                           species_id: Optional[int] = None) -> Data:
        """
        Build an atom graph without coordinate information used to infer topological
        structure of the molecule (optionally for a specific ``species_id``).

        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors to consider.
        residue_names : list of str, optional
            List of residue names to include in the template.
        atom_names : list of str, optional
            List of atom names to include in the template.
        species_id : int, optional
            Species identifier when dataset contains multiple molecular species.

        Returns
        -------
        Data
            Template graph.
        """
        if species_id is not None and hasattr(self, "species_structures") and species_id in self.species_structures:
            atomic_numbers = self.species_structures[species_id].get_atomic_numbers()
            bond_indices = self.species_bond_indices[species_id]
        else:
            atomic_numbers = self.structures[0].get_atomic_numbers()
            bond_indices = self.bond_indices
        return build_template_graph(atomic_numbers,
                                    bond_indices, 
                                    k=k, 
                                    residue_names=residue_names, 
                                    atom_names=atom_names,
                                    dtype=self.dtype)

    def get_species_template_graphs(
        self,
        k: int = 3,
        residue_names: Optional[List[str]] = None,
        atom_names: Optional[List[str]] = None,
    ) -> Dict[int, Data]:
        """
        Build template graphs for all distinct species in the dataset.

        Parameters
        ----------
        k : int, optional
            Maximum hop distance in the template graph.
        residue_names : list of str, optional
            Optional residue names.
        atom_names : list of str, optional
            Optional atom names.

        Returns
        -------
        dict of int to Data
            Mapping from ``species_id`` to its template graph.
        """
        templates: Dict[int, Data] = {}
        for sid in getattr(self, "species_structures", {}):
            templates[sid] = self.get_template_graph(
                k=k,
                residue_names=residue_names,
                atom_names=atom_names,
                species_id=sid,
            )
        return templates

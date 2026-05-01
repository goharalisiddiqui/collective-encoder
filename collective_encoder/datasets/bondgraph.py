import os
from multiprocessing import Pool
from typing import Any, List, Tuple, Dict, Optional, Union
import math

import numpy as np
import ase

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from .base import BaseDataset

from .utils.bondgraph import (assemble_bonds, bond_node_features, 
                              angle_and_torsion_edges,
                              build_template_graph)

from gslibs.utils.filesystem import create_rundir

# import metatensor.torch as mts
# import metatomic.torch as mta

# class MetatomicBondGraphDataset(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

     

#     def forward(
#         self,
#         systems: List[mta.System],
#         outputs: Dict[str, mta.ModelOutput],
#         selected_atoms: Optional[mts.Labels] = None,
#     ) -> torch.Tensor:

#         pass

def _compute_graph(structure: ase.Atoms, bond_indices: List[Tuple[int, int]]) -> Data:
    """Standalone graph computation from an ASE Atoms object and bond index list.
    Separated from the class so it can be called from worker processes without pickling self.
    """
    bonds = assemble_bonds(structure, bond_indices)
    x = bond_node_features(structure, bonds)
    edge_index, edge_attr, _, angle_atoms, torsion_atoms = angle_and_torsion_edges(structure, bonds)
    positions = torch.tensor(structure.get_positions())
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=positions,
        y_bonds=torch.tensor([b[2] for b in bonds]),
        y_angles=torch.tensor([edge_attr[i, 2]
                               for i in range(edge_attr.size(0)) if edge_attr[i, 0] == 1.0]),
        y_torsions_sin=torch.tensor([math.sin(edge_attr[i, 2])
                                     for i in range(edge_attr.size(0)) if edge_attr[i, 1] == 1.0]),
        y_torsions_cos=torch.tensor([math.cos(edge_attr[i, 2])
                                     for i in range(edge_attr.size(0)) if edge_attr[i, 1] == 1.0]),
    )


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
            graph.y = torch.tensor(label)
        torch.save(graph, os.path.join(graphs_dir, f"graph_{global_idx}.pt"))


class BondGraphDataset(BaseDataset, Dataset):
    """Graph dataset where nodes are bonds and edges connect bonds via angles or torsions.

    Node (bond) feature vector (length 3):
        [Z_i, Z_j, bond_length]

    Edge feature vector (length 3):
        [is_angle_edge, is_torsion_edge, value]
        value = bond angle (radians) for angle edges, dihedral angle (radians) for torsion edges.

    Edges are bidirectional (both directions added with same attributes).
    """

    _IDENTIFIER = "GRAPH"
    _REQUIRED_ARGS = ["bond_indices"]
    _OPTIONAL_ARGS = {
        'workdir': './.bge.tmp',
        'precompute_graphs': True,
        'parallel': True,
    }

    def __init__(
            self,
            structures: List[ase.Atoms],
            labels: Optional[List[float]] = None,
            dataset_args: Dict[str, Union[float, int, str]] = None,
            **kwargs,
        ):
        """Initialize dataset.

        bond_indices: list of (i,j) atom index pairs defining bonds (single global list applied to every structure).
        If validate_indices=True, will assert indices are in range for each structure.
        """
        BaseDataset.__init__(self, dataset_args=dataset_args, **kwargs)
        Dataset.__init__(self)
        
        self.structures = structures
        self.labels = labels
        
        self.bond_indices = [tuple(map(int, b)) for b in self.bond_indices]  # Ensure bond indices are tuples of ints
        # Validate bond indices
        n_atoms0 = len(structures[0])
        for (i, j) in self.bond_indices:
            assert 0 <= i < n_atoms0 and 0 <= j < n_atoms0, f"Bond index out of range: ({i},{j}) for n_atoms={n_atoms0}"

        self._calculate_label_indices()
        self.log_list("Bond set (atom index pairs)", self.bond_indices)
        self.log_list("Angle set (atom index triplets)", self.angle_index)
        self.log_list("Torsion set (atom index quartets)", self.torsion_index)
        
        if self.precompute_graphs:
            self.log_msg("Precomputing graph representations for all structures...")
            self._precompute_graphs()
        
        # Log dataset info
        self.log_msg(f"Loaded graph dataset with {self.len()} graphs") 
        self.log_msg(f"Number of bonds (nodes): {len(self.bond_indices)}") 
        sample = self.get(0)
        self.log_msg(f"Node feature size: {sample.x.shape[1]}") 
        self.log_msg(f"Edge feature size: {sample.edge_attr.shape[1]}") 
        self.log_msg(f"Total number of edges: {sample.edge_index.shape[1]}") 
    
    def _compute_graph(self, idx: int) -> Data:
        graph = _compute_graph(self.structures[idx], self.bond_indices)
        if self.labels is not None:
            graph.y = torch.tensor(self.labels[idx])
        return graph

    def _calculate_label_indices(self):
        atoms = self.structures[0]
        bonds = assemble_bonds(atoms, self.bond_indices)
        edge_index, edge_attr, _, angle_atoms, torsion_atoms = angle_and_torsion_edges(atoms, bonds)
        # Store bond <-> atom mapping
        self.bond_index = [[i, j] for (i, j, _) in bonds]
        self.angle_index = angle_atoms
        self.torsion_index = torsion_atoms
    
    def _precompute_graphs(self):
        precomputed_graphs_dir = create_rundir(self.workdir, "precomputed_graphs", 0, overwrite=False)
        os.makedirs(precomputed_graphs_dir, exist_ok=True)

        labels_list = self.labels if self.labels is not None else [None] * self.len()
        items = [(idx, struct, lbl)
                 for idx, (struct, lbl) in enumerate(zip(self.structures, labels_list))]

        if not self.parallel:
            for idx, structure, label in tqdm(items, desc="Precomputing graphs",
                                              disable=not self.verbose, dynamic_ncols=True):
                graph = _compute_graph(structure, self.bond_indices)
                if label is not None:
                    graph.y = torch.tensor(label)
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
    
    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)

    def get_datapoint_shape(self) -> Dict[str, Tuple]:
        """Return dictionary of data point tensor shapes."""
        sample = self.get(0)
        shapes = {
            'x': tuple(sample.x.shape),
            'edge_index': tuple(sample.edge_index.shape),
            'edge_attr': tuple(sample.edge_attr.shape),
            'pos': tuple(sample.pos.shape),
            'y_bonds': tuple(sample.y_bonds.shape),
            'y_angles': tuple(sample.y_angles.shape),
            'y_torsions_sin': tuple(sample.y_torsions_sin.shape),
            'y_torsions_cos': tuple(sample.y_torsions_cos.shape),
        }
        return shapes
    
    def get_norm_data(self) -> np.ndarray:
        """Return array of data to be normalized."""
        data_to_normalize = []
        for idx in range(self.len()):
            data = self.get(idx)
            node_feats = data.x.numpy()
            edge_feats = data.edge_attr.numpy()
            data_to_normalize.append(np.hstack([node_feats.mean(axis=0), edge_feats.mean(axis=0)]))
        data_to_normalize = np.vstack(data_to_normalize)
        return data_to_normalize
    
    def get_label_indices(self) -> List[int]:
        """Return list of atom indices of labels."""
        return self.bond_index, self.angle_index, self.torsion_index

    def len(self) -> int: # for PyG Dataset compatibility
        return len(self.structures)

    def get(self, idx: int) -> Data:  # for PyG Dataset compatibility
        if self.precompute_graphs:
            graph_path = os.path.join(self.precompute_graphs_dir, f"graph_{idx}.pt")
            graph = torch.load(graph_path)
        else:
            graph = self._compute_graph(idx)
        return graph
    
    def get_template_graph(self, 
                           k: int = 3,
                           residue_names: Optional[List[str]] = None,
                           atom_names: Optional[List[str]] = None) -> Data:
        atomic_numbers = self.structures[0].get_atomic_numbers()
        return build_template_graph(atomic_numbers,
                                    self.bond_indices, 
                                    k=k, 
                                    residue_names=residue_names, 
                                    atom_names=atom_names)

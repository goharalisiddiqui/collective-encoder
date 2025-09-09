from typing import List, Tuple, Dict, Optional
import math

import numpy as np
import ase
from ase.data import atomic_masses

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset

DOUBLE_PRECISION = False

def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return the signed dihedral angle (radians) for four points.
    Range (-pi, pi]."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    # Normalize b1 so that projection works correctly
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.arctan2(y, x))


class graphDataset(Dataset):
    """Graph dataset where nodes are bonds and edges connect bonds via angles or torsions.

    Node (bond) feature vector (length 6):
        [Z_i, Z_j, Z_i+Z_j, bond_length, mass_i, mass_j]
        (Interpreting "atomic number of the bond" as Z_i+Z_j.)

    Edge feature vector (length 3):
        [is_angle_edge, is_torsion_edge, value]
        value = bond angle (radians) for angle edges, dihedral angle (radians) for torsion edges.

    Edges are bidirectional (both directions added with same attributes).
    """

    def __init__(
            self,
            structures: List[ase.Atoms],
            bond_indices: List[Tuple[int, int]],
            labels: Optional[List[float]] = None,
            dtype: torch.dtype = torch.float32,
            add_torsion: bool = True,
            add_angles: bool = True,
            validate_indices: bool = True,
        ):
        """Initialize dataset.

        bond_indices: list of (i,j) atom index pairs defining bonds (single global list applied to every structure).
        If validate_indices=True, will assert indices are in range for each structure.
        """
        super().__init__()
        self.structures = structures
        self.labels = labels if labels is not None else [None] * len(structures)
        assert len(self.structures) == len(self.labels), "Structures and labels length mismatch"
        self.bond_indices: List[Tuple[int, int]] = [tuple(map(int, b)) for b in bond_indices]
        self.dtype = dtype
        self.add_torsion = add_torsion
        self.add_angles = add_angles
        if validate_indices:
            n_atoms0 = len(structures[0])
            for (i, j) in self.bond_indices:
                assert 0 <= i < n_atoms0 and 0 <= j < n_atoms0, f"Bond index out of range: ({i},{j}) for n_atoms={n_atoms0}"
    # ---------------------------------------------------------------------
    # Bond / feature construction helpers
    # ---------------------------------------------------------------------
    def _assemble_bonds(self, atoms: ase.Atoms) -> List[Tuple[int, int, float]]:
        """Compute bond lengths for the predefined bond index list."""
        pos = atoms.get_positions()
        bonds: List[Tuple[int, int, float]] = []
        for (i, j) in self.bond_indices:
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            bonds.append((i, j, dist))
        return bonds

    def _bond_node_features(self, atoms: ase.Atoms, bonds: List[Tuple[int, int, float]]) -> Tensor:
        Z = atoms.get_atomic_numbers()
        feats = []
        for i, j, d in bonds:
            Zi, Zj = Z[i], Z[j]
            # mi = atomic_masses[Zi]
            # mj = atomic_masses[Zj]
            feats.append([Zi, Zj, d])
        return torch.tensor(feats, dtype=self.dtype)

    def _angle_and_torsion_edges(self, atoms: ase.Atoms, bonds: List[Tuple[int, int, float]]):
        # Map bond (i,j) with i<j to bond index
        bond_index_map: Dict[Tuple[int, int], int] = {}
        for idx, (i, j, _) in enumerate(bonds):
            bond_index_map[(i, j)] = idx
            bond_index_map[(j, i)] = idx  # allow reverse lookup

        pos = atoms.get_positions()
        # Build adjacency: atom -> bonds containing it
        atom_to_bonds: Dict[int, List[int]] = {}
        for b_idx, (i, j, _) in enumerate(bonds):
            atom_to_bonds.setdefault(i, []).append(b_idx)
            atom_to_bonds.setdefault(j, []).append(b_idx)

        edge_index_src = []
        edge_index_dst = []
        edge_attr = []

        # Angle edges (adjacent bonds sharing one atom)
        if self.add_angles:
            for shared_atom, bond_list in atom_to_bonds.items():
                m = len(bond_list)
                if m < 2:
                    continue
                # For every unordered pair of bonds sharing this atom
                for a_i in range(m):
                    bi = bond_list[a_i]
                    ai1, ai2, _ = bonds[bi]
                    other_i = ai1 if ai2 == shared_atom else ai2
                    for a_j in range(a_i + 1, m):
                        bj = bond_list[a_j]
                        aj1, aj2, _ = bonds[bj]
                        other_j = aj1 if aj2 == shared_atom else aj2
                        # Compute angle at shared_atom: other_i - shared - other_j
                        v1 = pos[other_i] - pos[shared_atom]
                        v2 = pos[other_j] - pos[shared_atom]
                        n1 = np.linalg.norm(v1)
                        n2 = np.linalg.norm(v2)
                        if n1 == 0 or n2 == 0:
                            continue
                        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                        angle = float(np.arccos(cosang))
                        # Add both directions
                        for s, t in [(bi, bj), (bj, bi)]:
                            edge_index_src.append(s)
                            edge_index_dst.append(t)
                            edge_attr.append([1.0, 0.0, angle])

        # Torsion edges (bonds separated by exactly one bond: i-j and k-l with j-k bond existing)
        if self.add_torsion:
            # For torsion: for bond path (i,j)-(j,k)-(k,l) create edge between (i,j) and (k,l)
            # We'll iterate over central bond (j,k) and pairs of incident bonds on each side.
            # Build atom adjacency for quick bond existence test
            bonded_neighbors: Dict[int, List[int]] = {a: [] for a in range(len(atoms))}
            for i, j, _ in bonds:
                bonded_neighbors[i].append(j)
                bonded_neighbors[j].append(i)
            # Iterate central bond
            for (j, k, _) in bonds:
                lefts = [i for i in bonded_neighbors[j] if i != k]
                rights = [l for l in bonded_neighbors[k] if l != j]
                if not lefts or not rights:
                    continue
                for i in lefts:
                    # ensure bond (i,j) exists
                    if (i, j) not in bond_index_map:
                        continue
                    b_left = bond_index_map[(i, j)]
                    for l in rights:
                        if l in (i, j, k):
                            continue
                        if (k, l) not in bond_index_map:
                            continue
                        b_right = bond_index_map[(k, l)]
                        # dihedral i-j-k-l
                        angle = _dihedral(pos[i], pos[j], pos[k], pos[l])
                        angle = 0.5 + math.cos(angle - 1.25) # Following https://doi.org/10.1073/pnas.1600917113
                        for s, t in [(b_left, b_right), (b_right, b_left)]:
                            edge_index_src.append(s)
                            edge_index_dst.append(t)
                            edge_attr.append([0.0, 1.0, angle])

        if len(edge_index_src) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr_t = torch.empty((0, 3), dtype=self.dtype)
        else:
            edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            edge_attr_t = torch.tensor(edge_attr, dtype=self.dtype)

        return edge_index, edge_attr_t, bond_index_map

    # ------------------------------------------------------------------
    def len(self) -> int:  # for PyG Dataset compatibility
        return len(self.structures)

    def get(self, idx: int) -> Data:  # for PyG Dataset compatibility
        atoms = self.structures[idx]
        label = self.labels[idx]
        bonds = self._assemble_bonds(atoms)
        x = self._bond_node_features(atoms, bonds)
        edge_index, edge_attr, _ = self._angle_and_torsion_edges(atoms, bonds)
        # Store bond <-> atom mapping
        bond_atoms = torch.tensor([[i, j] for (i, j, _) in bonds], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, bond_index=bond_atoms)
        if label is not None:
            y = torch.tensor([label], dtype=self.dtype)
            data.y = y
        data.num_nodes = x.size(0)
        return data

    # torch.utils.data.Dataset interface (fallback)
    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)

__all__ = ["graphDataset"]

